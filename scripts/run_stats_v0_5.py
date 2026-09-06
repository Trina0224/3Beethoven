"""Fresh-base expanded curriculum training; fixed v0.3 recipe and new-family test."""
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
from pathlib import Path
from flight_run_stats_v0_3 import STUDENT,TEACHER,read_json,save_json,package
from stats_v0_3_common import prompt_for,digest,score_rows
from stats_curriculum_v0_5 import build
from run_stats_v0_4 import dataset,evaluate,BASE_REVISION,OLD_ADAPTER_SHA
from stats_holdout_v1 import questions as old_questions

ROOT=Path('/kaggle/working/3beethoven_stats_v0_5')


def examples(records):
    result=[]
    for r in records:
        for mode in ('letter','explain'):
            target=r['answer_letter'] if mode=='letter' else f"Answer: {r['answer_letter']}\n\nExplanation: {r['explanation']}\n\nCommon misconception: {r['common_mistake']}"
            result.append(dict(source_id=r['id'],family=r['family'],mode=mode,
                               prompt=prompt_for(r,mode),target=target))
    return result


def main():
    import torch
    import kagglehub
    from kaggle_secrets import UserSecretsClient
    from peft import PeftModel,LoraConfig,get_peft_model,prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,Trainer,TrainingArguments,set_seed
    from flight_run_stats_v0_1 import CausalCollator
    complete_generation=read_json(ROOT/'generation_complete.json')
    if not complete_generation: raise RuntimeError('Finish and preserve teacher generation before training')
    data=build()
    for split,qs in data.items():
        if read_json(ROOT/(split+'_questions.json'))!=qs: raise RuntimeError('Question freeze mismatch')
    train=read_json(ROOT/'train_records.json'); val=read_json(ROOT/'validation_records.json')
    for split,records in (('train',train),('validation',val)):
        if len(records)!=len(data[split]): raise RuntimeError('Incomplete teacher records')
        for r,q in zip(records,data[split]):
            if r['question_sha256']!=digest(q) or r['teacher_model']!=TEACHER: raise RuntimeError('Teacher provenance mismatch')
            review=read_json(ROOT/'reviews'/(r['review_tag'].removeprefix('review_')+'.json'))
            if not review or review.get('valid') is not True or review.get('answer_letter')!=q['answer_letter']:
                raise RuntimeError('Missing accepted content review')
    protocol=dict(epochs=3,learning_rate=5e-5,seed=226,effective_batch=8,lora_r=16,lora_alpha=32,
        lora_dropout=0.05,base_revision=BASE_REVISION,train_records_sha256=digest(train),
        validation_records_sha256=digest(val),test_sha256=digest(data['test']),
        training_sequences=360,validation_sequences=48,expected_optimizer_steps=135,
        primary='new-family four-rotation accuracy and all-four-correct',
        caveat='Expanded data and more optimizer steps than v0.3; not a compute-matched ablation')
    prev=read_json(ROOT/'training_protocol.json')
    if prev and prev!=protocol: raise RuntimeError('Training protocol mismatch')
    save_json(ROOT/'training_protocol.json',protocol)
    for name in ('run_stats_v0_5.py','run_stats_v0_4.py','run_stats_rotation_v1.py','flight_run_stats_v0_1.py','stats_holdout_v1.py','stats_holdout_v2.py'):
        shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    restored=Path(kagglehub.notebook_output_download('trinashih/3beethoven-v0-2/versions/5'))
    source=restored/'3beethoven_stats_flight_v0_3'
    adapter=source/'adapter'
    if not (adapter/'adapter_model.safetensors').exists():
        raise RuntimeError('Remove latest notebook input and mount version 5; never substitute another adapter')
    if hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest()!=OLD_ADAPTER_SHA:
        raise RuntimeError('Original adapter mismatch')
    if torch.cuda.device_count()!=1: raise RuntimeError('One visible GPU required')
    save_json(ROOT/'environment.json',dict(gpu=torch.cuda.get_device_name(0),packages={p:importlib.metadata.version(p) for p in ('torch','transformers','peft','bitsandbytes','datasets','accelerate')}))
    token=UserSecretsClient().get_secret('HF_TOKEN')
    tokenizer=AutoTokenizer.from_pretrained(adapter)
    def load_base():
        set_seed(226)
        model=AutoModelForCausalLM.from_pretrained(STUDENT,revision=BASE_REVISION,token=token,
            device_map={'':0},torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
        return prepare_model_for_kbit_training(model,gradient_checkpointing_kwargs={'use_reentrant':False})
    print('V05 TRAINING FROZEN',json.dumps(protocol),flush=True)
    summaries={}; suites=(('new',data['test']),('old',old_questions()))
    for label,qs in suites: save_json(ROOT/(label+'_benchmark.json'),qs)
    model=load_base()
    for name in ('baseline','v03'):
        if name=='v03': model=PeftModel.from_pretrained(model,adapter)
        summaries[name]={label:evaluate(model,tokenizer,qs,ROOT/f'{name}_{label}.json') for label,qs in suites}
        print('V05 MODEL',name,json.dumps({k:v['overall'] for k,v in summaries[name].items()}),flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()
    train_rows=examples(train); val_rows=examples(val)
    save_json(ROOT/'train_examples.json',train_rows); save_json(ROOT/'validation_examples.json',val_rows)
    model=load_base(); training=read_json(ROOT/'training_complete.json')
    if training:
        model=PeftModel.from_pretrained(model,ROOT/'adapter')
    else:
        model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,bias='none',task_type='CAUSAL_LM',
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
        model.config.use_cache=False
        args=TrainingArguments(output_dir=str(ROOT/'checkpoints'),num_train_epochs=3,
            per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=8,
            learning_rate=5e-5,warmup_steps=2,lr_scheduler_type='cosine',fp16=True,logging_steps=15,
            eval_strategy='epoch',save_strategy='epoch',save_total_limit=2,load_best_model_at_end=True,
            metric_for_best_model='eval_loss',greater_is_better=False,report_to='none',remove_unused_columns=False,
            optim='paged_adamw_8bit',seed=226)
        trainer=Trainer(model=model,args=args,train_dataset=dataset(train_rows,tokenizer),
            eval_dataset=dataset(val_rows,tokenizer),data_collator=CausalCollator(tokenizer))
        checkpoints=sorted((ROOT/'checkpoints').glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
        result=trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)
        model.save_pretrained(ROOT/'adapter'); tokenizer.save_pretrained(ROOT/'adapter')
        training=dict(training_loss=result.training_loss,best_validation_loss=trainer.state.best_metric,
            best_checkpoint=trainer.state.best_model_checkpoint,steps=trainer.state.global_step,
            train_questions=180,validation_questions=24,train_sequences=360,validation_sequences=48)
        save_json(ROOT/'training_complete.json',training); save_json(ROOT/'trainer_log.json',trainer.state.log_history)
        del trainer,model; gc.collect(); torch.cuda.empty_cache()
        model=PeftModel.from_pretrained(load_base(),ROOT/'adapter')
    summaries['v05']={label:evaluate(model,tokenizer,qs,ROOT/f'v05_{label}.json') for label,qs in suites}
    summary=dict(protocol=protocol,training=training,models=summaries,
        teacher_original_order=score_rows(read_json(ROOT/'teacher_test.json')),
        generation=complete_generation,adapter_sha256=hashlib.sha256((ROOT/'adapter'/'adapter_model.safetensors').read_bytes()).hexdigest())
    save_json(ROOT/'summary.json',summary); package(ROOT)
    archive=ROOT.parent/(ROOT.name+'.zip')
    print('V05 COMPLETE',json.dumps(summary),flush=True)
    print('V05 ARCHIVE',archive.stat().st_size,hashlib.sha256(archive.read_bytes()).hexdigest(),flush=True)


if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'; os.environ['TOKENIZERS_PARALLELISM']='false'
    main()
