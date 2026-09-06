"""Continue v0.10 using exact arithmetic supervision plus unchanged rehearsal."""
import os,json,gc,hashlib,shutil,importlib.metadata
from pathlib import Path
from flight_run_stats_v0_3 import STUDENT,read_json,save_json,package
from stats_v0_3_common import digest
from stats_curriculum_v0_11 import build,prompt,score
from run_stats_v0_4 import dataset,evaluate,BASE_REVISION
from stats_holdout_v1 import questions as old_questions
ROOT=Path('/kaggle/working/3beethoven_stats_v0_11')
SOURCE=Path('/kaggle/working/3beethoven_stats_v0_10')
SOURCE_SHA='14812770a7e612ab984e4ffad54bf514a3e00425655aa5adf732b975502f96f9'

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
        with torch.inference_mode():gen=model.generate(**inp,max_new_tokens=256,do_sample=False,pad_token_id=tok.eos_token_id)[0][inp['input_ids'].shape[-1]:]
        raw=tok.decode(gen,skip_special_tokens=True).strip()
        rows.append(dict(id=q['id'],category=q['category'],expected=q['answer'],prompt=p,raw=raw,generated_tokens=len(gen),hit_token_limit=len(gen)==256,**score(raw,q['answer'],q['category'])))
        save_json(path,rows)
        if len(rows)%12==0:print('V11 EVAL',path.stem,len(rows),'/',len(qs),flush=True)
    assert len(rows)==len(qs)
    def stats(rs):return dict(n=len(rs),strict=sum(r['correct'] for r in rs),reviewed=sum(r['reviewed_correct'] for r in rs),invalid=sum(r['invalid'] for r in rs),token_limit=sum(r['hit_token_limit'] for r in rs))
    return dict(overall=stats(rows),by_category={c:stats([r for r in rows if r['category']==c]) for c in sorted({r['category'] for r in rows})})

def main():
    import torch,kagglehub
    from kaggle_secrets import UserSecretsClient
    from peft import PeftModel,prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,Trainer,TrainingArguments,set_seed
    from flight_run_stats_v0_1 import CausalCollator
    repo=Path(__file__).resolve().parent.parent
    data=build();assert data==read_json(repo/'docs/STATS_V0_11_FROZEN_QUESTIONS.json')
    if not (SOURCE/'adapter/adapter_model.safetensors').exists():
        saved=Path(kagglehub.notebook_output_download('trinashih/3beethoven-v0-2/versions/18'))
        shutil.copytree(saved/SOURCE.name,SOURCE,dirs_exist_ok=True)
    assert hashlib.sha256((SOURCE/'adapter/adapter_model.safetensors').read_bytes()).hexdigest()==SOURCE_SHA
    ROOT.mkdir(exist_ok=True)
    rows={s:[dict(source_id=q['id'],mode='exact_arithmetic',prompt=prompt(q),target=q['target']) for q in data[s]]+read_json(SOURCE/(s+'_examples.json')) for s in ('train','validation')}
    assert len(rows['train'])==916 and len(rows['validation'])==104
    protocol=dict(data_sha256=digest(data),train_sha256=digest(rows['train']),validation_sha256=digest(rows['validation']),source_sha256=SOURCE_SHA,base_revision=BASE_REVISION,seed=1111,epochs=2,lr=3e-5,effective_batch=8,new_teacher_calls=0,method='Exact algorithmic arithmetic SFT plus existing Llama response-distillation rehearsal',numeric_max_tokens=256,primary='60 canonical arithmetic exact final values; variants and new statistical transfer separately')
    prev=read_json(ROOT/'training_protocol.json');assert prev is None or prev==protocol
    for name,obj in [('training_protocol',protocol),('frozen_questions',data),('train_examples',rows['train']),('validation_examples',rows['validation'])]:save_json(ROOT/(name+'.json'),obj)
    (ROOT/'source').mkdir(exist_ok=True)
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,ROOT/'source'/f.name)
    shutil.copy2(repo/'docs/STATS_V0_11_PROTOCOL.md',ROOT/'protocol.md')
    assert torch.cuda.device_count()==1
    token=UserSecretsClient().get_secret('HF_TOKEN');os.environ['HF_TOKEN']=token
    tok=AutoTokenizer.from_pretrained(SOURCE/'adapter')
    lengths=[len(tok.apply_chat_template([dict(role='user',content=r['prompt']),dict(role='assistant',content=r['target'])],tokenize=True,return_dict=False)) for rs in rows.values() for r in rs]
    assert max(lengths)<=768
    save_json(ROOT/'environment.json',dict(gpu=torch.cuda.get_device_name(0),max_sequence_tokens=max(lengths),packages={p:importlib.metadata.version(p) for p in ('torch','transformers','peft','bitsandbytes','datasets','accelerate')}))
    def base():
        set_seed(1111)
        m=AutoModelForCausalLM.from_pretrained(STUDENT,revision=BASE_REVISION,token=token,device_map={'':0},torch_dtype=torch.float16,quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
        return prepare_model_for_kbit_training(m,gradient_checkpointing_kwargs={'use_reentrant':False})
    training=read_json(ROOT/'training_complete.json')
    if not training:
        model=PeftModel.from_pretrained(base(),SOURCE/'adapter',is_trainable=True);model.config.use_cache=False
        args=TrainingArguments(output_dir=str(ROOT/'checkpoints'),num_train_epochs=2,per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=8,learning_rate=3e-5,warmup_steps=5,lr_scheduler_type='cosine',fp16=True,logging_steps=20,eval_strategy='epoch',save_strategy='epoch',save_total_limit=2,load_best_model_at_end=True,metric_for_best_model='eval_loss',greater_is_better=False,report_to='none',remove_unused_columns=False,optim='paged_adamw_8bit',seed=1111)
        trainer=Trainer(model=model,args=args,train_dataset=dataset(rows['train'],tok),eval_dataset=dataset(rows['validation'],tok),data_collator=CausalCollator(tok))
        cps=sorted((ROOT/'checkpoints').glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
        result=trainer.train(resume_from_checkpoint=str(cps[-1]) if cps else None)
        model.save_pretrained(ROOT/'adapter');tok.save_pretrained(ROOT/'adapter')
        training=dict(steps=trainer.state.global_step,best_checkpoint=trainer.state.best_model_checkpoint,best_validation_loss=trainer.state.best_metric,training_loss=result.training_loss)
        save_json(ROOT/'training_complete.json',training);save_json(ROOT/'trainer_log.json',trainer.state.log_history)
        package(ROOT);print('V11 TRAINED',json.dumps(training),flush=True)
        del trainer,model;gc.collect();torch.cuda.empty_cache()
    results={}
    for name,adapter in (('v10',SOURCE/'adapter'),('v11',ROOT/'adapter')):
        model=PeftModel.from_pretrained(base(),adapter)
        results[name]={label:numeric(model,tok,data[key],ROOT/f'{name}_{label}.json') for label,key in (('arithmetic','test'),('variants','test_variants'),('transfer','transfer'))}
        results[name]['old']=evaluate(model,tok,old_questions(),ROOT/f'{name}_old.json')
        save_json(ROOT/(name+'_metrics.json'),results[name]);print('V11 MODEL',name,json.dumps(results[name]),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    a,b=results['v10'],results['v11']
    summary=dict(protocol=protocol,training=training,models=results,adapter_sha256=hashlib.sha256((ROOT/'adapter/adapter_model.safetensors').read_bytes()).hexdigest(),goals=dict(arithmetic_gain_at_least_12=b['arithmetic']['overall']['reviewed']>=a['arithmetic']['overall']['reviewed']+12,arithmetic_at_least_half=b['arithmetic']['overall']['reviewed']>=30,transfer_gain_at_least_6=b['transfer']['overall']['reviewed']>=a['transfer']['overall']['reviewed']+6,retention=b['old']['overall']['correct']>=a['old']['overall']['correct']-4),format_review='Automated conservative final-scalar review; independent audit pending',new_teacher_calls=0)
    save_json(ROOT/'summary.json',summary);package(ROOT)
    archive=ROOT.with_suffix('.zip')
    print('V11 COMPLETE',json.dumps(summary),flush=True)
    print('V11 ARCHIVE',archive.stat().st_size,hashlib.sha256(archive.read_bytes()).hexdigest(),flush=True)

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['TOKENIZERS_PARALLELISM']='false';main()
