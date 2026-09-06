"""Joint replay repair with mandatory per-family retention gates."""
import os,gc,json,shutil
from pathlib import Path
from flight_run_stats_v0_3 import read_json as read,save_json as save,package,STUDENT
from run_stats_v0_17 import P15,HASHES,sha
from stats_curriculum_v0_19 import build,digest,score,prompt
ROOT=Path('/kaggle/working/3beethoven_stats_v0_19')
DATA_SHA='44935d14b7cc876b340ac92eb74ec5560229fe0227f34eaba1b930d5177d584f'
def metrics(rows):
    return dict(n=len(rows),correct=sum(r['correct'] for r in rows),pending=sum(r.get('review_required',False) for r in rows),by_category={c:sum(r['correct'] for r in rows if r['category']==c) for c in sorted({r['category'] for r in rows})})
def eligible(m,b):
    return m['correct']>=b['correct']-2 and all(m['by_category'][c]>=v-1 for c,v in b['by_category'].items())
def evaluate(model,tok,qs,path):
    import torch
    rows=read(path,[]);lookup={q['id']:q for q in qs}
    assert len({r['id'] for r in rows})==len(rows)
    for r in rows:
        q=lookup[r['id']];assert r['question_sha256']==digest(q) and r['prompt']==prompt(q)
        assert r['correct']==score(r['raw'],q)['correct']
    done={r['id'] for r in rows};was=model.training;cache=model.config.use_cache
    model.eval();model.config.use_cache=True
    for q in qs:
        if q['id'] in done:continue
        p=prompt(q);chat=tok.apply_chat_template([dict(role='user',content=p)],tokenize=False,add_generation_prompt=True)
        inputs=tok(chat,add_special_tokens=False,return_tensors='pt').to(model.device)
        with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=160,do_sample=False,pad_token_id=tok.eos_token_id)[0][inputs['input_ids'].shape[-1]:]
        raw=tok.decode(out,skip_special_tokens=True).strip()
        rows.append(dict(id=q['id'],category=q['category'],question_sha256=digest(q),prompt=p,raw=raw,generated_tokens=len(out),hit_token_limit=len(out)==160,**score(raw,q)))
        save(path,rows)
    model.config.use_cache=cache;model.train(was)
    return metrics(rows)
def main():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,Trainer,TrainingArguments,TrainerCallback,set_seed
    from peft import PeftModel,prepare_model_for_kbit_training
    from kaggle_secrets import UserSecretsClient
    from run_stats_v0_4 import BASE_REVISION,dataset
    from flight_run_stats_v0_1 import CausalCollator
    repo=Path(__file__).resolve().parents[1];ROOT.mkdir(exist_ok=True)
    data=build();assert digest(data)==DATA_SHA and data==read(repo/'docs/STATS_V0_19_FROZEN_QUESTIONS.json')
    assert sha(P15/'adapter/adapter_model.safetensors')==HASHES[15]
    save(ROOT/'frozen_questions.json',data);shutil.copy2(repo/'docs/STATS_V0_19_PROTOCOL.md',ROOT/'protocol.md')
    source=ROOT/'source';source.mkdir(exist_ok=True)
    for p in (repo/'scripts').glob('*.py'):shutil.copy2(p,source/p.name)
    save(ROOT/'provenance.json',dict(data_sha256=DATA_SHA,v15_sha256=HASHES[15],base_revision=BASE_REVISION,new_teacher_calls=0,old_teacher_rows=192,procedural_rows=288))
    token=UserSecretsClient().get_secret('HF_TOKEN');tok=AutoTokenizer.from_pretrained(P15/'adapter')
    def load(path,training=False):
        set_seed(1919)
        model=AutoModelForCausalLM.from_pretrained(STUDENT,revision=BASE_REVISION,token=token,device_map={'':0},torch_dtype=torch.float16,quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
        if training:model=prepare_model_for_kbit_training(model,gradient_checkpointing_kwargs={'use_reentrant':False})
        model=PeftModel.from_pretrained(model,path,is_trainable=training);model.config.use_cache=not training;return model
    model=load(P15/'adapter');baseline={k:evaluate(model,tok,data[k],ROOT/('v15_'+k+'.json')) for k in ('old_validation','new_validation')}
    save(ROOT/'baseline.json',baseline);print('V19 BASELINE',baseline,flush=True)
    del model;gc.collect();torch.cuda.empty_cache()
    if not (ROOT/'training_complete.json').exists():
        model=load(P15/'adapter',True)
        class Monitor(TrainerCallback):
            def on_epoch_end(self,args,state,control,**kwargs):
                epoch=int(round(state.epoch));folder=ROOT/f'epoch_{epoch}';folder.mkdir(exist_ok=True)
                m={k:evaluate(model,tok,data[k],folder/(k+'.json')) for k in ('old_validation','new_validation')}
                ok=eligible(m['old_validation'],baseline['old_validation'])
                record=dict(epoch=epoch,step=state.global_step,metrics=m,eligible=ok)
                history=[r for r in read(ROOT/'history.json',[]) if r['epoch']<epoch];history.append(record)
                best=max(history,key=lambda r:(r['metrics']['new_validation']['correct'],r['metrics']['old_validation']['correct'],-r['epoch']))
                reason='retention_gate_failed' if not ok else ('validation_plateau' if epoch-best['epoch']>=2 else ('budget_cap' if epoch==4 else None))
                record['stop_reason']=reason;save(ROOT/'history.json',history)
                model.save_pretrained(folder/'adapter');tok.save_pretrained(folder/'adapter')
                save(folder/'adapter_hash.json',dict(sha256=sha(folder/'adapter/adapter_model.safetensors')))
                print('V19 EPOCH',json.dumps(record),flush=True)
                control.should_save=True
                if reason:control.should_training_stop=True
                return control
        args=TrainingArguments(output_dir=str(ROOT/'checkpoints'),num_train_epochs=4,per_device_train_batch_size=1,gradient_accumulation_steps=8,learning_rate=2e-5,lr_scheduler_type='constant',warmup_steps=0,fp16=True,logging_steps=15,save_strategy='epoch',report_to='none',remove_unused_columns=False,optim='paged_adamw_8bit',seed=1919,disable_tqdm=True)
        trainer=Trainer(model=model,args=args,train_dataset=dataset(data['train'],tok),data_collator=CausalCollator(tok),callbacks=[Monitor()])
        ckpts=sorted((ROOT/'checkpoints').glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
        result=trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)
        save(ROOT/'training_complete.json',dict(steps=trainer.state.global_step,loss=result.training_loss,logs=trainer.state.log_history))
        del trainer,model;gc.collect();torch.cuda.empty_cache()
    history=read(ROOT/'history.json');allowed=[r for r in history if r['eligible']]
    selected=max(allowed,key=lambda r:(r['metrics']['new_validation']['correct'],r['metrics']['old_validation']['correct'],-r['epoch'])) if allowed else None
    selection=dict(selected_epoch=selected['epoch'] if selected else None,diagnostic_epoch=history[-1]['epoch'] if not selected else None,rule='Validation only; no candidate passing retention keeps v15.')
    prior=read(ROOT/'selection.json');assert prior is None or prior==selection;save(ROOT/'selection.json',selection)
    candidate=selected['epoch'] if selected else history[-1]['epoch'];paths={'v15':P15/'adapter',f'epoch_{candidate}':ROOT/f'epoch_{candidate}'/'adapter'}
    save(ROOT/'test_candidates.json',{k:str(v) for k,v in paths.items()});tests={}
    for name,path in paths.items():
        model=load(path);tests[name]={k:evaluate(model,tok,data[k],ROOT/(name+'_'+k+'.json')) for k in ('old_test','new_test')}
        del model;gc.collect();torch.cuda.empty_cache();save(ROOT/'test_metrics.json',tests);print('V19 TEST',name,tests[name],flush=True)
    c=tests[f'epoch_{candidate}'];b=tests['v15']
    success=bool(selected) and c['new_test']['correct']>=b['new_test']['correct']+8 and c['old_test']['correct']>=b['old_test']['correct']-4 and all(c['old_test']['by_category'][k]>=v-2 for k,v in b['old_test']['by_category'].items())
    save(ROOT/'summary.json',dict(selection=selection,baseline=baseline,history=history,tests=tests,automatic_success=success,pending=sum(m['pending'] for t in tests.values() for m in t.values()),note='Manual review required before final promotion; no old MC claim.'))
    package(ROOT);print('V19 COMPLETE',flush=True)
if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['TOKENIZERS_PARALLELISM']='false';main()
