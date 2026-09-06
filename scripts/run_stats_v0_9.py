"""Fresh-base concise solution training and separately frozen comparisons."""
import os,gc,json,hashlib,shutil,importlib.metadata
from pathlib import Path
from flight_run_stats_v0_3 import STUDENT,TEACHER,read_json,save_json,package
from stats_curriculum_v0_9 import build,prompt
from stats_v0_3_common import digest,prompt_for,parse_answer
from run_stats_v0_4 import dataset,evaluate,BASE_REVISION
from diagnose_stats_v0_7 import ADAPTER_SHA,score
from generate_stats_v0_9 import ROOT,Client,validate,target,validated_solution
from stats_holdout_v1 import questions as old_questions

def examples(records):
    out=[]
    for r in records:
        out.append(dict(source_id=r['id'],mode='letter',prompt=prompt_for(r),target=r['answer_letter']))
        out.append(dict(source_id=r['id'],mode='numeric',prompt=prompt(r),target=r['target']))
    return out

def numeric_evaluate(model,tok,questions,path):
    import torch
    rows=read_json(path,[]);done={r['id'] for r in rows}
    model.eval();model.config.use_cache=True
    for q in questions:
        if q['id'] in done:continue
        question_prompt=prompt(q)
        chat=tok.apply_chat_template([dict(role='user',content=question_prompt)],tokenize=False,add_generation_prompt=True)
        inp=tok(chat,add_special_tokens=False,return_tensors='pt').to(model.device)
        with torch.inference_mode():gen=model.generate(**inp,max_new_tokens=256,do_sample=False,pad_token_id=tok.eos_token_id)[0][inp['input_ids'].shape[-1]:]
        raw=tok.decode(gen,skip_special_tokens=True).strip()
        pred,correct,invalid=score(raw,dict(mode='numeric',expected=q['answer']))
        rows.append(dict(id=q['id'],category=q['category'],prompt=question_prompt,expected=q['answer'],raw=raw,predicted=pred,correct=correct,invalid=invalid,generated_tokens=len(gen),hit_token_limit=len(gen)==256))
        save_json(path,rows)
        if len(rows)%12==0:print('V09 NUMERIC',path.stem,len(rows),'/48',flush=True)
    return dict(n=len(rows),correct=sum(r['correct'] for r in rows),invalid=sum(r['invalid'] for r in rows),hit_token_limit=sum(r['hit_token_limit'] for r in rows))

def main():
    import torch,kagglehub
    from kaggle_secrets import UserSecretsClient
    from peft import PeftModel,LoraConfig,get_peft_model,prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,Trainer,TrainingArguments,set_seed
    from flight_run_stats_v0_1 import CausalCollator
    data=build();assert data==read_json(Path(__file__).resolve().parent.parent/'docs/STATS_V0_9_FROZEN_QUESTIONS.json')
    train=read_json(ROOT/'train_records.json');val=read_json(ROOT/'validation_records.json')
    audit=read_json(ROOT/'audit_approved.json')
    assert audit and audit['approved'] is True and audit['records_sha256']==digest(dict(train=train,validation=val))
    for split,rows in (('train',train),('validation',val)):
        assert len(rows)==len(data[split])
        for r,q in zip(rows,data[split]):
            assert all(r[k]==v for k,v in q.items())
            obj=validate(r['teacher_solution'],q);assert r['target']==target(obj)
            cached=read_json(ROOT/'api_cache'/(r['cache_tag']+'.json'))
            from stats_v0_3_common import parse_teacher
            assert obj==validated_solution(cached['text'],q)
    protocol=dict(base_revision=BASE_REVISION,data_sha256=digest(data),train_sha256=digest(train),validation_sha256=digest(val),epochs=3,steps=135,seed=226,lr=5e-5,effective_batch=8,lora_r=16,lora_alpha=32,lora_dropout=.05,train_sequences=360,validation_sequences=48,primary='new MC accuracy, double baseline, all-four correct',secondary='new no-choice compact numerical solutions',limits='Targeted curriculum and format change together; new instances of taught skills, not unseen-family transfer')
    prior=read_json(ROOT/'training_protocol.json');assert prior is None or prior==protocol
    save_json(ROOT/'training_protocol.json',protocol)
    (ROOT/'source').mkdir(exist_ok=True)
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,ROOT/'source'/f.name)
    shutil.copy2(Path(__file__).resolve().parent.parent/'docs/STATS_V0_9_PROTOCOL.md',ROOT/'protocol.md')
    saved=Path(kagglehub.notebook_output_download('trinashih/3beethoven-v0-2/versions/9'));adapter=saved/'3beethoven_stats_v0_5/adapter'
    assert hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest()==ADAPTER_SHA
    assert torch.cuda.device_count()==1
    token=UserSecretsClient().get_secret('HF_TOKEN');os.environ['HF_TOKEN']=token
    tok=AutoTokenizer.from_pretrained(adapter)
    def base():
        set_seed(226)
        m=AutoModelForCausalLM.from_pretrained(STUDENT,revision=BASE_REVISION,token=token,device_map={'':0},torch_dtype=torch.float16,quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
        return prepare_model_for_kbit_training(m,gradient_checkpointing_kwargs={'use_reentrant':False})
    save_json(ROOT/'environment.json',dict(gpu=torch.cuda.get_device_name(0),packages={p:importlib.metadata.version(p) for p in ('torch','transformers','peft','bitsandbytes','datasets','accelerate')}))
    summaries={};model=base()
    for name in ('baseline','v05'):
        if name=='v05':model=PeftModel.from_pretrained(model,adapter)
        summaries[name]=dict(mc=evaluate(model,tok,data['test'],ROOT/(name+'_mc.json')),numeric=numeric_evaluate(model,tok,data['test'],ROOT/(name+'_numeric.json')))
        print('V09 MODEL',name,json.dumps(dict(mc=summaries[name]['mc']['overall'],numeric=summaries[name]['numeric'])),flush=True)
    del model;gc.collect();torch.cuda.empty_cache()
    train_rows=examples(train);val_rows=examples(val)
    save_json(ROOT/'train_examples.json',train_rows);save_json(ROOT/'validation_examples.json',val_rows)
    model=base();training=read_json(ROOT/'training_complete.json')
    if training:model=PeftModel.from_pretrained(model,ROOT/'adapter')
    else:
        model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
        model.config.use_cache=False
        args=TrainingArguments(output_dir=str(ROOT/'checkpoints'),num_train_epochs=3,per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=8,learning_rate=5e-5,warmup_steps=2,lr_scheduler_type='cosine',fp16=True,logging_steps=15,eval_strategy='epoch',save_strategy='epoch',save_total_limit=2,load_best_model_at_end=True,metric_for_best_model='eval_loss',greater_is_better=False,report_to='none',remove_unused_columns=False,optim='paged_adamw_8bit',seed=226)
        trainer=Trainer(model=model,args=args,train_dataset=dataset(train_rows,tok),eval_dataset=dataset(val_rows,tok),data_collator=CausalCollator(tok))
        checkpoints=sorted((ROOT/'checkpoints').glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
        result=trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)
        model.save_pretrained(ROOT/'adapter');tok.save_pretrained(ROOT/'adapter')
        training=dict(steps=trainer.state.global_step,best_checkpoint=trainer.state.best_model_checkpoint,best_validation_loss=trainer.state.best_metric,training_loss=result.training_loss)
        save_json(ROOT/'training_complete.json',training);save_json(ROOT/'trainer_log.json',trainer.state.log_history)
        del trainer,model;gc.collect();torch.cuda.empty_cache();model=PeftModel.from_pretrained(base(),ROOT/'adapter')
    summaries['v09']=dict(mc=evaluate(model,tok,data['test'],ROOT/'v09_mc.json'),numeric=numeric_evaluate(model,tok,data['test'],ROOT/'v09_numeric.json'),old=evaluate(model,tok,old_questions(),ROOT/'v09_old.json'))
    print('V09 MODEL v09',json.dumps(dict(mc=summaries['v09']['mc']['overall'],numeric=summaries['v09']['numeric'],old=summaries['v09']['old']['overall'])),flush=True)
    teacher=read_json(ROOT/'teacher_test.json',[]);done={r['id'] for r in teacher}
    client=Client(UserSecretsClient().get_secret('OPENROUTER_API_KEY'))
    for q in data['test']:
        if q['id'] in done:continue
        raw=client.call('test_'+q['id'],[dict(role='user',content=prompt_for(q))],max_tokens=16)
        pred=parse_answer(raw);teacher.append(dict(id=q['id'],category=q['category'],raw=raw,predicted=pred,expected=q['answer_letter'],correct=pred==q['answer_letter']))
        save_json(ROOT/'teacher_test.json',teacher)
    accuracy=summaries['v09']['mc']['overall']['accuracy']
    goals=dict(at_least_60_percent=accuracy>=.6,double_baseline=accuracy>=2*summaries['baseline']['mc']['overall']['accuracy'],half_questions_all_four=summaries['v09']['mc']['all_four_correct']>=24)
    summary=dict(protocol=protocol,training=training,models=summaries,goals=goals,teacher_original=dict(n=48,correct=sum(r['correct'] for r in teacher)),usage=client.stats(),adapter_sha256=hashlib.sha256((ROOT/'adapter/adapter_model.safetensors').read_bytes()).hexdigest())
    save_json(ROOT/'summary.json',summary);save_json(ROOT/'api_usage.json',client.stats());package(ROOT)
    print('V09 COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['TOKENIZERS_PARALLELISM']='false';main()
