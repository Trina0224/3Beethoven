"""Depth curriculum with replay, stage checkpoints and matched shuffled exposure."""
import os, gc, json, shutil, random
from pathlib import Path
from collections import Counter
from flight_run_stats_v0_3 import save_json as save, read_json as read, package, STUDENT
from run_stats_v0_17 import restore, P15, sha, HASHES
from stats_curriculum_v0_18 import build, digest, prompt, score, stage_rows, TRACKS

ROOT=Path('/kaggle/working/3beethoven_stats_v0_18')
DATA_SHA='0e51ed04578b29e8d91f931dfbeec78a94867ba3d52dd6a45e06c45ee186f57c'
SEED=1818

def metrics(rows):
    def count(rs):return dict(n=len(rs),correct=sum(r['correct'] for r in rs),pending=sum(r['review_required'] for r in rs))
    return dict(**count(rows),by_depth={str(d):count([r for r in rows if r['depth']==d]) for d in (1,2,3)},
                by_cell={f'{t}_d{d}':count([r for r in rows if r['depth']==d and r['track']==t]) for t in TRACKS for d in (1,2,3)})

def evaluate(model,tok,qs,path):
    import torch
    rows=read(path,[]);lookup={q['id']:q for q in qs}
    assert len({r['id'] for r in rows})==len(rows)
    for r in rows:assert r['question_sha256']==digest(lookup[r['id']]) and r['prompt']==prompt(lookup[r['id']])
    done={r['id'] for r in rows};was_training=model.training;old_cache=model.config.use_cache
    model.eval();model.config.use_cache=True
    for q in qs:
        if q['id'] in done:continue
        p=prompt(q);chat=tok.apply_chat_template([dict(role='user',content=p)],tokenize=False,add_generation_prompt=True)
        inputs=tok(chat,add_special_tokens=False,return_tensors='pt').to(model.device)
        with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=160,do_sample=False,pad_token_id=tok.eos_token_id)[0][inputs['input_ids'].shape[-1]:]
        raw=tok.decode(out,skip_special_tokens=True).strip()
        rows.append(dict(id=q['id'],track=q['track'],depth=q['depth'],question_sha256=digest(q),prompt=p,raw=raw,generated_tokens=len(out),hit_token_limit=len(out)==160,**score(raw,q)))
        save(path,rows)
    model.config.use_cache=old_cache;model.train(was_training)
    return metrics(rows)

def stage_key(m,stage):
    active=m['by_depth'][str(stage)]['correct']
    lower=sum(m['by_depth'][str(d)]['correct'] for d in range(1,stage))
    weakest=min(m['by_cell'][f'{t}_d{stage}']['correct'] for t in TRACKS)
    return active,lower,weakest

def mastery(m,stage):
    return (m['by_depth'][str(stage)]['correct']>=14 and
            all(m['by_cell'][f'{t}_d{stage}']['correct']>=3 for t in TRACKS) and
            all(m['by_depth'][str(d)]['correct']>=12 for d in range(1,stage)))

def main():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,Trainer,TrainingArguments,TrainerCallback,set_seed
    from peft import PeftModel,prepare_model_for_kbit_training
    from kaggle_secrets import UserSecretsClient
    from run_stats_v0_4 import BASE_REVISION,dataset
    from flight_run_stats_v0_1 import CausalCollator
    ROOT.mkdir(exist_ok=True);repo=Path(__file__).resolve().parents[1];data=build()
    assert digest(data)==DATA_SHA and data==read(repo/'docs/STATS_V0_18_FROZEN_QUESTIONS.json')
    protocol=dict(data_sha256=DATA_SHA,source_v15_sha256=HASHES[15],seed=SEED,lr=2e-5,
        max_epochs_per_stage=8,min_epochs_per_stage=3,plateau_patience=3,mastery_consecutive_checks=2,
        convergence='After at least 3 epochs: two consecutive mastery checks OR 3 epochs without lexicographic validation improvement. Plateau below mastery is explicitly labeled, not called learning success. At the 8-epoch budget cap without convergence, stop progression.',
        mastery='Current depth >=14/16 and each track >=3/4; each previous depth >=12/16',
        stage_sizes=[96,144,192],replay='Stage 2: 96 depth-2 +48 depth-1. Stage 3:96 depth-3 +48 depth-2 +48 depth-1.',
        continuation='Last converged epoch, no rollback; full checkpoint at every epoch, explicit boundary adapter.',
        selection='Stage endpoints only, frozen before final test; best validation epochs recorded diagnostically but never selected by test.',
        control='Global shuffle of the exact realized training multiset, divided at matching optimizer-reset boundaries; same initialization, LR, updates and batch size. Budget determined by curriculum validation. One seed; does not isolate adaptive budget selection.',
        target_provenance='Procedural exact supervised references, not new Llama teacher responses; curriculum ablation within the statistics pilot.',new_teacher_calls=0)
    prior=read(ROOT/'protocol.json');assert prior is None or prior==protocol
    save(ROOT/'protocol.json',protocol);save(ROOT/'frozen_questions.json',data)
    source=ROOT/'source';source.mkdir(exist_ok=True)
    for p in (repo/'scripts').glob('*.py'):shutil.copy2(p,source/p.name)
    shutil.copy2(repo/'docs/STATS_V0_18_PROTOCOL.md',ROOT/'protocol.md')
    restore();token=UserSecretsClient().get_secret('HF_TOKEN');tok=AutoTokenizer.from_pretrained(P15/'adapter')
    def load(path,training=False):
        set_seed(SEED)
        model=AutoModelForCausalLM.from_pretrained(STUDENT,revision=BASE_REVISION,token=token,device_map={'':0},torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
        if training:model=prepare_model_for_kbit_training(model,gradient_checkpointing_kwargs={'use_reentrant':False})
        model=PeftModel.from_pretrained(model,path,is_trainable=training);model.config.use_cache=not training
        return model
    model=load(P15/'adapter');baseline=evaluate(model,tok,data['validation'],ROOT/'baseline_validation.json')
    save(ROOT/'baseline_metrics.json',baseline);print('V18 BASELINE',baseline,flush=True)
    del model;gc.collect();torch.cuda.empty_cache()
    stages=[];previous=P15/'adapter'
    for stage in (1,2,3):
        folder=ROOT/f'stage_{stage}';folder.mkdir(exist_ok=True)
        saved=read(folder/'complete.json')
        if saved:
            stages.append(saved);previous=folder/'adapter'
            if not saved['converged']:break
            continue
        rows=stage_rows(data,stage);save(folder/'train.json',rows);model=load(previous,True)
        class Monitor(TrainerCallback):
            def on_epoch_end(self,args,state,control,**kwargs):
                epoch=int(round(state.epoch));m=evaluate(model,tok,data['validation'],folder/f'epoch_{epoch}_validation.json')
                history=read(folder/'history.json',[])
                history=[r for r in history if r['epoch']<epoch]
                history.append(dict(epoch=epoch,step=state.global_step,metrics=m,mastery=mastery(m,stage)))
                best=max(range(len(history)),key=lambda i:stage_key(history[i]['metrics'],stage))
                stable=len(history)>=2 and all(r['mastery'] for r in history[-2:])
                plateau=len(history)-1-best>=3
                reason=('stable_mastery' if stable else 'validation_plateau') if epoch>=3 and (stable or plateau) else None
                history[-1]['stop_reason']=reason;save(folder/'history.json',history)
                print('V18 STAGE',stage,'EPOCH',epoch,'DEPTH',m['by_depth'],'STOP',reason,flush=True)
                control.should_save=True
                if reason:control.should_training_stop=True
                return control
        args=TrainingArguments(output_dir=str(folder/'checkpoints'),num_train_epochs=8,per_device_train_batch_size=1,gradient_accumulation_steps=8,
            learning_rate=2e-5,lr_scheduler_type='constant',warmup_steps=0,fp16=True,logging_steps=12,save_strategy='epoch',
            report_to='none',remove_unused_columns=False,optim='paged_adamw_8bit',seed=SEED+stage,disable_tqdm=True)
        trainer=Trainer(model=model,args=args,train_dataset=dataset(rows,tok),data_collator=CausalCollator(tok),callbacks=[Monitor()])
        ckpts=sorted((folder/'checkpoints').glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
        result=trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)
        model.save_pretrained(folder/'adapter');tok.save_pretrained(folder/'adapter')
        history=read(folder/'history.json');last=history[-1];reason=last['stop_reason'] or 'budget_cap_without_convergence'
        record=dict(stage=stage,epochs=last['epoch'],steps=trainer.state.global_step,converged=bool(last['stop_reason']),stop_reason=reason,
                    mastered=last['mastery'],validation=last['metrics'],adapter_sha256=sha(folder/'adapter/adapter_model.safetensors'),loss=result.training_loss,log=trainer.state.log_history)
        save(folder/'complete.json',record);stages.append(record);previous=folder/'adapter'
        del trainer,model;gc.collect();torch.cuda.empty_cache()
        package(ROOT);print('V18 STAGE COMPLETE',stage,reason,flush=True)
        if not record['converged']:break
    save(ROOT/'stage_summary.json',stages)
    # Replay order control: every training row appears exactly as many times.
    exposure=[];lengths=[]
    for s in stages:
        chunk=stage_rows(data,s['stage'])*s['epochs'];exposure+=chunk;lengths.append(len(chunk))
    shuffled=list(exposure);random.Random(SEED).shuffle(shuffled)
    assert Counter(r['source_id'] for r in exposure)==Counter(r['source_id'] for r in shuffled)
    save(ROOT/'control_exposure.json',shuffled);save(ROOT/'curriculum_exposure.json',exposure)
    controls=[];previous=P15/'adapter';offset=0
    for stage,n in enumerate(lengths,1):
        folder=ROOT/f'control_{stage}';folder.mkdir(exist_ok=True);chunk=shuffled[offset:offset+n];offset+=n
        completed=read(folder/'complete.json')
        if completed:controls.append(completed);previous=folder/'adapter';continue
        model=load(previous,True)
        args=TrainingArguments(output_dir=str(folder/'checkpoints'),num_train_epochs=1,per_device_train_batch_size=1,gradient_accumulation_steps=8,
            learning_rate=2e-5,lr_scheduler_type='constant',warmup_steps=0,fp16=True,logging_steps=12,save_strategy='steps',save_steps=12,save_total_limit=2,
            report_to='none',remove_unused_columns=False,optim='paged_adamw_8bit',seed=SEED+stage,disable_tqdm=True)
        trainer=Trainer(model=model,args=args,train_dataset=dataset(chunk,tok),data_collator=CausalCollator(tok))
        ckpts=sorted((folder/'checkpoints').glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
        result=trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)
        assert trainer.state.global_step==stages[stage-1]['steps']
        model.save_pretrained(folder/'adapter');tok.save_pretrained(folder/'adapter')
        m=evaluate(model,tok,data['validation'],folder/'validation.json')
        completed=dict(stage=stage,steps=trainer.state.global_step,validation=m,adapter_sha256=sha(folder/'adapter/adapter_model.safetensors'),loss=result.training_loss)
        save(folder/'complete.json',completed);controls.append(completed);previous=folder/'adapter'
        del trainer,model;gc.collect();torch.cuda.empty_cache();print('V18 CONTROL COMPLETE',stage,m['by_depth'],flush=True)
    paths={'v15':P15/'adapter',**{f'stage_{s["stage"]}':ROOT/f'stage_{s["stage"]}'/'adapter' for s in stages},f'control_{len(stages)}':previous}
    save(ROOT/'test_candidates.json',dict(paths={n:str(p) for n,p in paths.items()},selection='All curriculum boundaries and final shuffled control, declared before test'))
    tests={}
    for name,path in paths.items():
        model=load(path);tests[name]=evaluate(model,tok,data['test'],ROOT/(name+'_test.json'))
        save(ROOT/'test_metrics.json',tests);print('V18 TEST',name,tests[name],flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    save(ROOT/'summary.json',dict(protocol=protocol,baseline=baseline,stages=stages,controls=controls,tests=tests))
    package(ROOT);print('V18 COMPLETE',flush=True)

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['TOKENIZERS_PARALLELISM']='false';main()
