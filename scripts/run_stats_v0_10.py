"""Continue v0.9 with staged fractions and rehearsal; freeze and retain raw tests."""
import os,gc,json,hashlib,shutil,importlib.metadata
from pathlib import Path
from flight_run_stats_v0_3 import STUDENT,read_json,save_json,package
from stats_curriculum_v0_10 import build,prompt
from stats_v0_3_common import digest,prompt_for,parse_answer,parse_teacher
from run_stats_v0_4 import dataset,evaluate,BASE_REVISION
from run_stats_v0_9 import examples,numeric_evaluate
from stats_holdout_v1 import questions as old_questions
from generate_stats_v0_10 import ROOT,Client,validate
SOURCE=Path('/kaggle/working/3beethoven_stats_v0_9')
ADAPTER_SHA='805a2170a805f6176aa3837857890b8c44fc8f854d16cbc3085ae220e5502c7c'

def prepare():
    repo=Path(__file__).resolve().parent.parent;data=build()
    assert data==read_json(repo/'docs/STATS_V0_10_FROZEN_QUESTIONS.json')
    records=read_json(ROOT/'teacher_records.json');audit=read_json(ROOT/'audit_approved.json')
    assert audit and audit['approved'] and audit['records_sha256']==digest(records)
    rules=read_json(SOURCE/'rules.json')
    assert digest(rules)==read_json(SOURCE/'rules_approved.json')['rules_sha256']
    rows={}
    for split in ('train','validation'):
        rows[split]=[]
        for r,q in zip(records[split],data[split]):
            assert all(r[k]==v for k,v in q.items())
            obj=validate(r['teacher_solution'],q)
            assert obj==parse_teacher(read_json(ROOT/'api_cache'/(r['cache_tag']+'.json'))['text'])
            text='Formula: '+rules[q['category']]['rule']+'\nCalculation: '+' = '.join(obj['stages'])+'\nAnswer: '+obj['answer']
            rows[split].append(dict(source_id=q['id'],mode='staged_fraction',prompt=prompt(q),target=text))
        assert len(rows[split])==len(data[split])
        rows[split]+=examples(read_json(SOURCE/(split+'_records.json')))
    prior=read_json(repo/'docs/STATS_V0_5_FROZEN_QUESTIONS.json')
    for topic in ('poisson','expectation','uniform','type_i','type_ii','confidence'):
        selected=[q for q in prior['train'] if q['category']==topic][:10]
        assert len(selected)==10
        rows['train'] += [dict(source_id=q['id'],mode='rehearsal_letter',prompt=prompt_for(q),target=q['answer_letter']) for q in selected]
    assert len(rows['train'])==516 and len(rows['validation'])==64
    for s,r in rows.items():save_json(ROOT/(s+'_examples.json'),r)
    return data,rows

def main():
    import torch
    from kaggle_secrets import UserSecretsClient
    from peft import PeftModel,prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,Trainer,TrainingArguments,set_seed
    from flight_run_stats_v0_1 import CausalCollator
    data,rows=prepare()
    assert hashlib.sha256((SOURCE/'adapter/adapter_model.safetensors').read_bytes()).hexdigest()==ADAPTER_SHA
    protocol=dict(data_sha256=digest(data),train_sha256=digest(rows['train']),validation_sha256=digest(rows['validation']),source_adapter_sha256=ADAPTER_SHA,base_revision=BASE_REVISION,seed=1010,epochs=2,lr=2e-5,effective_batch=8)
    prior=read_json(ROOT/'training_protocol.json');assert prior is None or prior==protocol
    save_json(ROOT/'training_protocol.json',protocol)
    (ROOT/'source').mkdir(exist_ok=True)
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,ROOT/'source'/f.name)
    shutil.copy2(Path(__file__).resolve().parent.parent/'docs/STATS_V0_10_PROTOCOL.md',ROOT/'protocol.md')
    token=UserSecretsClient().get_secret('HF_TOKEN');os.environ['HF_TOKEN']=token
    tok=AutoTokenizer.from_pretrained(SOURCE/'adapter')
    lengths=[len(tok(tok.apply_chat_template([dict(role='user',content=r['prompt']),dict(role='assistant',content=r['target'])],tokenize=False),add_special_tokens=False)['input_ids']) for rs in rows.values() for r in rs]
    assert max(lengths)<=768
    def base():
        set_seed(1010)
        m=AutoModelForCausalLM.from_pretrained(STUDENT,revision=BASE_REVISION,token=token,device_map={'':0},torch_dtype=torch.float16,quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
        return prepare_model_for_kbit_training(m,gradient_checkpointing_kwargs={'use_reentrant':False})
    assert torch.cuda.device_count()==1
    save_json(ROOT/'environment.json',dict(gpu=torch.cuda.get_device_name(0),max_sequence_tokens=max(lengths),packages={p:importlib.metadata.version(p) for p in ('torch','transformers','peft','bitsandbytes','datasets','accelerate')}))
    results={};model=base()
    for name in ('baseline','v09'):
        if name=='v09':model=PeftModel.from_pretrained(model,SOURCE/'adapter')
        results[name]=dict(mc=evaluate(model,tok,data['test'],ROOT/(name+'_mc.json')),numeric=numeric_evaluate(model,tok,data['test'],ROOT/(name+'_numeric.json')))
        if name=='v09':results[name]['old']=evaluate(model,tok,old_questions(),ROOT/'v09_old.json')
        print('V10 MODEL',name,json.dumps({k:(v['overall'] if 'overall' in v else v) for k,v in results[name].items()}),flush=True)
    del model;gc.collect();torch.cuda.empty_cache()
    training=read_json(ROOT/'training_complete.json')
    model=PeftModel.from_pretrained(base(),ROOT/'adapter' if training else SOURCE/'adapter',is_trainable=not bool(training))
    if not training:
        model.config.use_cache=False
        args=TrainingArguments(output_dir=str(ROOT/'checkpoints'),num_train_epochs=2,per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=8,learning_rate=2e-5,warmup_steps=2,lr_scheduler_type='cosine',fp16=True,logging_steps=15,eval_strategy='epoch',save_strategy='epoch',save_total_limit=2,load_best_model_at_end=True,metric_for_best_model='eval_loss',greater_is_better=False,report_to='none',remove_unused_columns=False,optim='paged_adamw_8bit',seed=1010)
        trainer=Trainer(model=model,args=args,train_dataset=dataset(rows['train'],tok),eval_dataset=dataset(rows['validation'],tok),data_collator=CausalCollator(tok))
        cps=sorted((ROOT/'checkpoints').glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
        result=trainer.train(resume_from_checkpoint=str(cps[-1]) if cps else None)
        model.save_pretrained(ROOT/'adapter');tok.save_pretrained(ROOT/'adapter')
        training=dict(steps=trainer.state.global_step,best_checkpoint=trainer.state.best_model_checkpoint,best_validation_loss=trainer.state.best_metric,training_loss=result.training_loss)
        save_json(ROOT/'training_complete.json',training);save_json(ROOT/'trainer_log.json',trainer.state.log_history)
        del trainer,model;gc.collect();torch.cuda.empty_cache();model=PeftModel.from_pretrained(base(),ROOT/'adapter')
    results['v10']=dict(mc=evaluate(model,tok,data['test'],ROOT/'v10_mc.json'),numeric=numeric_evaluate(model,tok,data['test'],ROOT/'v10_numeric.json'),old=evaluate(model,tok,old_questions(),ROOT/'v10_old.json'))
    print('V10 MODEL v10',json.dumps({k:(v['overall'] if 'overall' in v else v) for k,v in results['v10'].items()}),flush=True)
    client=Client(UserSecretsClient().get_secret('OPENROUTER_API_KEY'));teacher=read_json(ROOT/'teacher_test.json',[]);done={r['id'] for r in teacher}
    for q in data['test']:
        if q['id'] in done:continue
        raw=client.call('test_'+q['id'],[dict(role='user',content=prompt_for(q))],max_tokens=16)
        pred=parse_answer(raw);teacher.append(dict(id=q['id'],raw=raw,predicted=pred,expected=q['answer_letter'],correct=pred==q['answer_letter']))
        save_json(ROOT/'teacher_test.json',teacher)
    summary=dict(protocol=protocol,training=training,models=results,teacher_original=dict(n=48,correct=sum(r['correct'] for r in teacher)),usage=client.stats(),adapter_sha256=hashlib.sha256((ROOT/'adapter/adapter_model.safetensors').read_bytes()).hexdigest(),goals_strict=dict(numeric_gain_eight=results['v10']['numeric']['correct']>=results['v09']['numeric']['correct']+8,numeric_at_least_half=results['v10']['numeric']['correct']>=24,retention=results['v10']['old']['overall']['correct']>=results['v09']['old']['overall']['correct']-4),primary_format_review='Independent review pending; strict flags above are not final reviewed primary outcomes')
    save_json(ROOT/'summary.json',summary);save_json(ROOT/'api_usage.json',client.stats());package(ROOT)
    print('V10 COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['TOKENIZERS_PARALLELISM']='false';main()
