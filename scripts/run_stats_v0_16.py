"""Train through three support levels, then evaluate frozen paired questions."""
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
from pathlib import Path
from flight_run_stats_v0_3 import save_json as save,read_json as read,package,STUDENT
from stats_curriculum_v0_16 import build,digest,prompt,score
from review_teacher_v0_16 import teacher_score

ROOT=Path('/kaggle/working/3beethoven_stats_v0_16')
PREV=Path('/kaggle/working/3beethoven_stats_v0_15')
SOURCE_SHA='9369d52de4a886df9da0c872cd41bd4e01af0a38bf02ad724b5951c1a6b9f5d3'
DATA_SHA='db2a2a3513e680ca64cea2d2a80e84f23e332377b140bcbc5150ccd0311c0f6c'

def restore():
    target=PREV/'adapter/adapter_model.safetensors'
    if not target.exists():
        import kagglehub
        cache=Path(kagglehub.notebook_output_download('trinashih/3beethoven-v0-2/versions/29'))
        candidates=[p for p in cache.rglob('adapter_model.safetensors') if p.parent.parent.name==PREV.name]
        assert candidates,'Saved v15 adapter not found'
        shutil.copytree(candidates[0].parent,PREV/'adapter',dirs_exist_ok=True)
    assert hashlib.sha256(target.read_bytes()).hexdigest()==SOURCE_SHA

def examples(data,repo):
    records=read(ROOT/'teacher/records.json');out={}
    for split in ('train','validation'):
        rs={r['id']:r for r in records[split]};out[split]=[]
        for q in data[split]:
            r=rs[q['id']];assert r['question_sha256']==digest(q)
            for a in r['attempts']:
                g=teacher_score(a['raw'],q)
                if g['correct']:
                    out[split].append(dict(q=q,target='Expression: '+g['normalized_expression'],teacher_raw=a['raw'],attempt=a['attempt']));break
    assert len(out['train'])>=180 and len(out['validation'])>=30
    for task in ('mean','variance','moment','scale'):
        assert sum(r['q']['task']==task for r in out['train'])>=42,task+' coverage'
    old=read(repo/'docs/STATS_V0_14_VERIFIED_DISTILLATION.json')['train'];replay=[]
    for kind in ('poisson_time','poisson_scaled','moment','uniform_time','binomial','exactly_one','at_least_one','interval'):
        subset=[r for r in old if f'_{kind}_' in r['source_id']][:6]
        assert len(subset)==6;replay.extend(subset)
    return out,replay

def evaluate_formulas(model,tok,qs,path,support):
    import torch
    rows=read(path,[]);lookup={q['id']:q for q in qs}
    assert len({r['id'] for r in rows})==len(rows)
    for r in rows:assert r['question_sha256']==digest(lookup[r['id']]) and r['prompt']==prompt(lookup[r['id']],support)
    done={r['id'] for r in rows};model.eval();model.config.use_cache=True
    for q in qs:
        if q['id'] in done:continue
        p=prompt(q,support)
        chat=tok.apply_chat_template([dict(role='user',content=p)],tokenize=False,add_generation_prompt=True)
        inputs=tok(chat,add_special_tokens=False,return_tensors='pt').to(model.device)
        with torch.inference_mode():
            generated=model.generate(**inputs,max_new_tokens=160,do_sample=False,pad_token_id=tok.eos_token_id)[0][inputs['input_ids'].shape[-1]:]
        raw=tok.decode(generated,skip_special_tokens=True).strip()
        rows.append(dict(id=q['id'],category=q['category'],question_sha256=digest(q),prompt=p,raw=raw,
            generated_tokens=len(generated),hit_token_limit=len(generated)==160,**score(raw,q)))
        save(path,rows)
        if len(rows)%12==0:print('V16 EVAL',path.stem,len(rows),'/',len(qs),flush=True)
    return dict(n=len(rows),correct=sum(r['correct'] for r in rows),pending=sum(r['review_required'] for r in rows),
        by_category={c:dict(n=sum(r['category']==c for r in rows),correct=sum(r['correct'] for r in rows if r['category']==c)) for c in sorted({q['category'] for q in qs})})

def main():
    import torch
    from kaggle_secrets import UserSecretsClient
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,Trainer,TrainingArguments,set_seed
    from peft import PeftModel,prepare_model_for_kbit_training
    from flight_run_stats_v0_1 import CausalCollator
    from run_stats_v0_4 import BASE_REVISION,dataset,evaluate
    from stats_holdout_v1 import questions as old_questions
    data=build();repo=Path(__file__).resolve().parents[1]
    assert digest(data)==DATA_SHA and data==read(repo/'docs/STATS_V0_16_FROZEN_QUESTIONS.json')
    restore();ROOT.mkdir(exist_ok=True)
    save(ROOT/'frozen_questions.json',data)
    source=ROOT/'source';source.mkdir(exist_ok=True)
    for p in (repo/'scripts').glob('*.py'):shutil.copy2(p,source/p.name)
    corpus,replay=examples(data,repo)
    validation=[dict(prompt=prompt(r['q']),target=r['target']) for r in corpus['validation']]
    tok=AutoTokenizer.from_pretrained(PREV/'adapter');token=UserSecretsClient().get_secret('HF_TOKEN')
    protocol=dict(data_sha256=DATA_SHA,source_sha256=SOURCE_SHA,seed=1616,base_revision=BASE_REVISION,
        stages=[dict(name='full',epochs=1,lr=2e-5),dict(name='cue',epochs=1,lr=2e-5),dict(name='none',epochs=3,lr=2e-5)],
        replay_count=len(replay),new_train=len(corpus['train']),validation=len(validation),
        selection='Unaided validation loss; best checkpoint within final no-hint stage. No test selection.',
        method='Verified Llama teacher responses; same-story contrasts; fading support; response distillation, not logits',
        test='96 fresh problem identities in known template families; separate 24 task-specific-rule aided probes; old 240 rotations')
    previous=read(ROOT/'protocol.json');assert previous is None or previous==protocol
    save(ROOT/'protocol.json',protocol)
    save(ROOT/'environment.json',dict(gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        packages={p:importlib.metadata.version(p) for p in ('torch','transformers','peft','bitsandbytes','datasets','accelerate')}))
    def base(training=False):
        set_seed(1616)
        m=AutoModelForCausalLM.from_pretrained(STUDENT,revision=BASE_REVISION,token=token,device_map={'':0},
            torch_dtype=torch.float16,quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
        return prepare_model_for_kbit_training(m,gradient_checkpointing_kwargs={'use_reentrant':False}) if training else m
    adapter=PREV/'adapter'
    for spec in protocol['stages']:
        name=spec['name'];stage=ROOT/name;complete=read(stage/'complete.json')
        if not complete:
            train=[dict(prompt=prompt(r['q'],name),target=r['target'],source_id=r['q']['id'],teacher_raw=r['teacher_raw']) for r in corpus['train']]+replay
            save(stage/'train.json',train);save(stage/'validation.json',validation)
            model=PeftModel.from_pretrained(base(True),adapter,is_trainable=True);model.config.use_cache=False
            args=TrainingArguments(output_dir=str(stage/'checkpoints'),num_train_epochs=spec['epochs'],
                per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=8,
                learning_rate=spec['lr'],warmup_steps=3,lr_scheduler_type='cosine',fp16=True,logging_steps=10,
                eval_strategy='epoch',save_strategy='epoch',save_total_limit=2,load_best_model_at_end=True,
                metric_for_best_model='eval_loss',greater_is_better=False,report_to='none',remove_unused_columns=False,
                optim='paged_adamw_8bit',seed=1616)
            trainer=Trainer(model=model,args=args,train_dataset=dataset(train,tok),eval_dataset=dataset(validation,tok),data_collator=CausalCollator(tok))
            ckpts=sorted((stage/'checkpoints').glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
            result=trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)
            model.save_pretrained(stage/'adapter');tok.save_pretrained(stage/'adapter')
            complete=dict(steps=trainer.state.global_step,best_checkpoint=trainer.state.best_model_checkpoint,
                best_validation_loss=trainer.state.best_metric,train_loss=result.training_loss)
            save(stage/'complete.json',complete);save(stage/'log.json',trainer.state.log_history)
            print('V16 TRAINED STAGE',name,complete,flush=True)
            del trainer,model;gc.collect();torch.cuda.empty_cache()
        adapter=stage/'adapter'
    shutil.copytree(adapter,ROOT/'adapter',dirs_exist_ok=True)
    package(ROOT);print('V16 STUDENT TRAINED',flush=True)
    results={};weak=[q for q in data['test'] if q['category'] in ('moment','poisson_scaled')]
    for name,adapter in (('v15',PREV/'adapter'),('v16',ROOT/'adapter')):
        model=PeftModel.from_pretrained(base(),adapter)
        results[name]=dict(unaided=evaluate_formulas(model,tok,data['test'],ROOT/(name+'_unaided.json'),'none'),
            aided=evaluate_formulas(model,tok,weak,ROOT/(name+'_aided.json'),'full'),
            retention=evaluate(model,tok,old_questions(),ROOT/(name+'_old.json')))
        save(ROOT/(name+'_metrics.json'),results[name]);print('V16 MODEL',name,results[name],flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    summary=dict(protocol=protocol,models=results,teacher=read(ROOT/'teacher/summary.json'),
        training={s['name']:read(ROOT/s['name']/'complete.json') for s in protocol['stages']},
        adapter_sha256=hashlib.sha256((ROOT/'adapter/adapter_model.safetensors').read_bytes()).hexdigest())
    save(ROOT/'summary.json',summary);package(ROOT)
    from verify_stats_v0_16 import main as verify
    verify()

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['TOKENIZERS_PARALLELISM']='false';main()
