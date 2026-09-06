"""Train and evaluate the alternative procedural-arithmetic experiment."""
import gc,hashlib,importlib.metadata,json,os,shutil
from pathlib import Path
from flight_run_stats_v0_3 import STUDENT,read_json,save_json,package
from flight_run_stats_v0_1 import CausalCollator
from run_stats_v0_4 import BASE_REVISION,dataset,evaluate
from stats_holdout_v1 import questions as old_questions
from stats_curriculum_v0_12 import build,prompt,score
from stats_v0_3_common import digest

ROOT=Path('/kaggle/working/3beethoven_stats_v0_12')
V10=Path('/kaggle/working/3beethoven_stats_v0_10')
V11=Path('/kaggle/working/3beethoven_stats_v0_11')
V10_SHA='14812770a7e612ab984e4ffad54bf514a3e00425655aa5adf732b975502f96f9'
V11_SHA='9994b0eb73cf824791ffbeb81dd08a301bb08801e2c38f38829af3cfd8618541'

def restore():
    if all((p/'adapter/adapter_model.safetensors').exists() for p in (V10,V11)):return
    import kagglehub
    saved=Path(kagglehub.notebook_output_download('trinashih/3beethoven-v0-2/versions/24'))
    for dest in (V10,V11):
        if (dest/'adapter/adapter_model.safetensors').exists():continue
        candidates=sorted((p for p in saved.rglob(dest.name) if p.is_dir()),key=lambda p:len(p.parts))
        assert candidates,(saved,dest.name)
        shutil.copytree(candidates[0],dest,dirs_exist_ok=True)
    assert hashlib.sha256((V10/'adapter/adapter_model.safetensors').read_bytes()).hexdigest()==V10_SHA
    assert hashlib.sha256((V11/'adapter/adapter_model.safetensors').read_bytes()).hexdigest()==V11_SHA

def numeric(model,tok,qs,path):
    import torch
    rows=read_json(path,[]);lookup={q['id']:q for q in qs}
    assert len({r['id'] for r in rows})==len(rows)
    for r in rows:assert r['expected']==lookup[r['id']]['answer'] and r['prompt']==prompt(lookup[r['id']])
    done={r['id'] for r in rows};model.eval();model.config.use_cache=True
    for q in qs:
        if q['id'] in done:continue
        p=prompt(q);chat=tok.apply_chat_template([dict(role='user',content=p)],tokenize=False,add_generation_prompt=True)
        inp=tok(chat,add_special_tokens=False,return_tensors='pt').to(model.device)
        with torch.inference_mode():gen=model.generate(**inp,max_new_tokens=192,do_sample=False,pad_token_id=tok.eos_token_id)[0][inp['input_ids'].shape[-1]:]
        raw=tok.decode(gen,skip_special_tokens=True).strip();judged=score(raw,q['answer'],q['category'])
        rows.append(dict(id=q['id'],category=q['category'],expected=q['answer'],prompt=p,raw=raw,generated_tokens=len(gen),hit_token_limit=len(gen)==192,**judged));save_json(path,rows)
        if len(rows)%16==0:print('V12 EVAL',path.stem,len(rows),'/',len(qs),flush=True)
    def stats(rs):return dict(n=len(rs),correct=sum(r['correct'] for r in rs),invalid=sum(r['invalid'] for r in rs),token_limit=sum(r['hit_token_limit'] for r in rs))
    return dict(overall=stats(rows),by_category={c:stats([r for r in rows if r['category']==c]) for c in sorted({r['category'] for r in rows})})

def main():
    import torch
    from kaggle_secrets import UserSecretsClient
    from peft import PeftModel,prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,Trainer,TrainingArguments,set_seed
    repo=Path(__file__).resolve().parent.parent;restore();data=build()
    assert data==read_json(repo/'docs/STATS_V0_12_FROZEN_QUESTIONS.json')
    ROOT.mkdir(exist_ok=True)
    rows={s:[dict(source_id=q['id'],mode='procedural_arithmetic',prompt=prompt(q),target=q['target']) for q in data[s]]+read_json(V10/(s+'_examples.json')) for s in ('train','validation')}
    assert len(rows['train'])==1076 and len(rows['validation'])==128
    protocol=dict(data_sha256=digest(data),train_sha256=digest(rows['train']),validation_sha256=digest(rows['validation']),source='v0.10',source_sha256=V10_SHA,comparison_v11_sha256=V11_SHA,base_revision=BASE_REVISION,seed=1212,epochs=2,lr=2e-5,effective_batch=8,new_teacher_calls=0,method='Procedural arithmetic SFT plus unchanged audited Llama response-distillation rehearsal',numeric_max_tokens=192,checkpoint_selection='validation loss only')
    prev=read_json(ROOT/'training_protocol.json');assert prev is None or prev==protocol
    for name,obj in [('training_protocol',protocol),('frozen_questions',data),('train_examples',rows['train']),('validation_examples',rows['validation'])]:save_json(ROOT/(name+'.json'),obj)
    (ROOT/'source').mkdir(exist_ok=True)
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,ROOT/'source'/f.name)
    shutil.copy2(repo/'docs/STATS_V0_12_PROTOCOL.md',ROOT/'protocol.md')
    assert torch.cuda.device_count()>=1
    token=UserSecretsClient().get_secret('HF_TOKEN');os.environ['HF_TOKEN']=token;tok=AutoTokenizer.from_pretrained(V10/'adapter')
    lengths=[len(tok.apply_chat_template([dict(role='user',content=r['prompt']),dict(role='assistant',content=r['target'])],tokenize=True,return_dict=False)) for rs in rows.values() for r in rs]
    assert max(lengths)<=768
    save_json(ROOT/'environment.json',dict(gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],max_sequence_tokens=max(lengths),packages={p:importlib.metadata.version(p) for p in ('torch','transformers','peft','bitsandbytes','datasets','accelerate')}))
    def base():
        set_seed(1212)
        m=AutoModelForCausalLM.from_pretrained(STUDENT,revision=BASE_REVISION,token=token,device_map={'':0},torch_dtype=torch.float16,quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
        return prepare_model_for_kbit_training(m,gradient_checkpointing_kwargs={'use_reentrant':False})
    training=read_json(ROOT/'training_complete.json')
    if not training:
        model=PeftModel.from_pretrained(base(),V10/'adapter',is_trainable=True);model.config.use_cache=False
        args=TrainingArguments(output_dir=str(ROOT/'checkpoints'),num_train_epochs=2,per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=8,learning_rate=2e-5,warmup_steps=5,lr_scheduler_type='cosine',fp16=True,logging_steps=20,eval_strategy='epoch',save_strategy='epoch',save_total_limit=2,load_best_model_at_end=True,metric_for_best_model='eval_loss',greater_is_better=False,report_to='none',remove_unused_columns=False,optim='paged_adamw_8bit',seed=1212)
        trainer=Trainer(model=model,args=args,train_dataset=dataset(rows['train'],tok),eval_dataset=dataset(rows['validation'],tok),data_collator=CausalCollator(tok))
        cps=sorted((ROOT/'checkpoints').glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]));result=trainer.train(resume_from_checkpoint=str(cps[-1]) if cps else None)
        model.save_pretrained(ROOT/'adapter');tok.save_pretrained(ROOT/'adapter');training=dict(steps=trainer.state.global_step,best_checkpoint=trainer.state.best_model_checkpoint,best_validation_loss=trainer.state.best_metric,training_loss=result.training_loss)
        save_json(ROOT/'training_complete.json',training);save_json(ROOT/'trainer_log.json',trainer.state.log_history);package(ROOT);print('V12 TRAINED',json.dumps(training),flush=True)
        del trainer,model;gc.collect();torch.cuda.empty_cache()
    results={}
    for name,adapter in (('v10',V10/'adapter'),('v11',V11/'adapter'),('v12',ROOT/'adapter')):
        model=PeftModel.from_pretrained(base(),adapter)
        results[name]={label:numeric(model,tok,data[key],ROOT/f'{name}_{label}.json') for label,key in (('micro','micro_test'),('transfer','transfer_test'))}
        results[name]['old']=evaluate(model,tok,old_questions(),ROOT/f'{name}_old.json');save_json(ROOT/(name+'_metrics.json'),results[name]);print('V12 MODEL',name,json.dumps(results[name]),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    a,b=results['v10'],results['v12']
    goals=dict(micro_gain_at_least_16=b['micro']['overall']['correct']>=a['micro']['overall']['correct']+16,micro_at_least_48=b['micro']['overall']['correct']>=48,transfer_gain_at_least_8=b['transfer']['overall']['correct']>=a['transfer']['overall']['correct']+8,transfer_at_least_24=b['transfer']['overall']['correct']>=24,retention=b['old']['overall']['correct']>=a['old']['overall']['correct']-4)
    summary=dict(protocol=protocol,training=training,models=results,adapter_sha256=hashlib.sha256((ROOT/'adapter/adapter_model.safetensors').read_bytes()).hexdigest(),goals=goals,full_success=all(goals.values()),new_teacher_calls=0,review='Independent raw-output review pending')
    save_json(ROOT/'summary.json',summary);package(ROOT);archive=ROOT.with_suffix('.zip')
    print('V12 COMPLETE',json.dumps(summary),flush=True);print('V12 ARCHIVE',archive.stat().st_size,hashlib.sha256(archive.read_bytes()).hexdigest(),flush=True)

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['TOKENIZERS_PARALLELISM']='false';main()
