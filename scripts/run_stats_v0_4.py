"""Bounded, zero-teacher-call position repair; fresh base, 36 optimizer steps."""
import hashlib
import importlib.metadata
import json
import os
import shutil
from pathlib import Path
from collections import Counter
from flight_run_stats_v0_3 import STUDENT, read_json, read_jsonl, save_json, package
from stats_v0_3_common import group_split, prompt_for, parse_answer, digest, score_rows, make_curriculum
from stats_holdout_v1 import questions as old_questions
from stats_holdout_v2 import questions as new_questions, validate
from run_stats_rotation_v1 import rotate, original_index

ROOT=Path('/kaggle/working/3beethoven_stats_v0_4')
BASE_REVISION='0cb88a4f764b7a12671c53f0838cd831a0843b95'
OLD_ADAPTER_SHA='7c3dd4513bd4f9e98ae03b9788f60a5337689de20056936cf03f7dba02bed4cf'
CONFIG=dict(max_steps=36,learning_rate=5e-5,batch_size=1,gradient_accumulation_steps=8,
            seed=226,max_length=768,lora_r=16,lora_alpha=32,lora_dropout=0.05)


def examples(records):
    result=[]
    for r in records:
        for shift in range(4):
            changed=rotate(r,shift)
            result.append(dict(source_id=r['id'],mode='letter',shift=shift,
                prompt=prompt_for(changed),target=changed['answer_letter']))
        result.append(dict(source_id=r['id'],mode='explain',shift=0,
            prompt=prompt_for(r,'explain'),target=f"Answer: {r['answer_letter']}\n\nExplanation: {r['explanation']}\n\nCommon misconception: {r['common_mistake']}"))
    return result


def dataset(rows,tokenizer):
    from datasets import Dataset
    result=[]
    for r in rows:
        messages=[dict(role='user',content=r['prompt'])]
        prefix=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
        full=tokenizer.apply_chat_template(messages+[dict(role='assistant',content=r['target'])],tokenize=True,add_generation_prompt=False,return_dict=False)
        if full[:len(prefix)]!=prefix or not len(prefix)<len(full)<=CONFIG['max_length']:
            raise RuntimeError('Unsafe supervision boundary or truncation')
        result.append(dict(input_ids=full,attention_mask=[1]*len(full),labels=[-100]*len(prefix)+full[len(prefix):]))
    return Dataset.from_list(result)


def metrics(rows,questions):
    assert len(rows)==4*len(questions)
    assert {(r['id'],r['shift']) for r in rows}=={(q['id'],s) for q in questions for s in range(4)}
    details=[]
    for q in questions:
        g=sorted([r for r in rows if r['id']==q['id']],key=lambda r:r['shift'])
        choices=[r['original_choice_index'] for r in g]
        details.append(dict(id=q['id'],category=q['category'],correct_rotations=sum(r['correct'] for r in g),
                            semantic_consistency=None not in choices and len(set(choices))==1))
    return dict(overall=score_rows(rows),all_four_correct=sum(r['correct_rotations']==4 for r in details),
        all_four_wrong=sum(r['correct_rotations']==0 for r in details),
        semantically_consistent=sum(r['semantic_consistency'] for r in details),
        by_gold_position={k:score_rows([r for r in rows if r['expected']==k]) for k in 'ABCD'},
        by_category={k:score_rows([r for r in rows if r['category']==k]) for k in sorted({q['category'] for q in questions})},
        hit_token_limit=sum(r['hit_token_limit'] for r in rows),question_details=details)


def evaluate(model,tokenizer,questions,path):
    import torch
    rows=read_json(path,[]); done={(r['id'],r['shift']) for r in rows}
    model.eval(); model.config.use_cache=True
    for q in questions:
        for shift in range(4):
            if (q['id'],shift) in done: continue
            changed=rotate(q,shift)
            chat=tokenizer.apply_chat_template([dict(role='user',content=prompt_for(changed))],tokenize=False,add_generation_prompt=True)
            inputs=tokenizer(chat,add_special_tokens=False,return_tensors='pt').to(model.device)
            with torch.inference_mode():
                out=model.generate(**inputs,max_new_tokens=16,do_sample=False,pad_token_id=tokenizer.eos_token_id)
            generated=out[0][inputs['input_ids'].shape[-1]:]
            raw=tokenizer.decode(generated,skip_special_tokens=True).strip(); pred=parse_answer(raw)
            rows.append(dict(id=q['id'],category=q['category'],shift=shift,expected=changed['answer_letter'],
                predicted=pred,correct=pred==changed['answer_letter'],raw=raw,
                original_choice_index=original_index(pred,shift),generated_tokens=len(generated),hit_token_limit=len(generated)==16))
            save_json(path,rows)
            if len(rows)%48==0: print('V04 EVAL',path.stem,len(rows),'/',4*len(questions),flush=True)
    return metrics(rows,questions)


def main():
    import torch
    from kaggle_secrets import UserSecretsClient
    from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    from flight_run_stats_v0_1 import CausalCollator
    import kagglehub
    source_mount=Path(kagglehub.notebook_output_download('trinashih/3beethoven-v0-2/versions/5'))
    candidates=[p for p in source_mount.rglob('teacher_train.jsonl') if (p.parent/'adapter'/'adapter_model.safetensors').exists()]
    if len(candidates)!=1: raise RuntimeError('Mount original version 5 output with corpus and adapter')
    source=candidates[0].parent
    if hashlib.sha256((source/'adapter'/'adapter_model.safetensors').read_bytes()).hexdigest()!=OLD_ADAPTER_SHA:
        raise RuntimeError('Unexpected original adapter')
    records=read_jsonl(candidates[0]); reference={r['id']:r for r in make_curriculum()}
    if len(records)!=60 or len({r['id'] for r in records})!=60: raise RuntimeError('Corpus incomplete')
    for r in records:
        if r['teacher_model']!='meta-llama/llama-3.3-70b-instruct': raise RuntimeError('Unexpected teacher')
        if any(r[k]!=reference[r['id']][k] for k in ('question','choices','answer_letter')): raise RuntimeError('Corpus reference mismatch')
    train,val=group_split(records)
    original_split=read_json(source/'split.json')
    assert set(original_split['train_ids'])=={r['id'] for r in train}
    assert set(original_split['validation_ids'])=={r['id'] for r in val}
    fresh=new_questions(); old=old_questions(); audit=validate(records+old)
    ROOT.mkdir(exist_ok=True)
    protocol=dict(config=CONFIG,training_corpus_sha256=digest(records),new_benchmark=audit,
        old_benchmark_sha256=digest(old),teacher_calls=0,base_revision=BASE_REVISION,
        training='Fresh base; four rotated letter targets plus one unchanged explanation per question',
        limits='Compound change in augmentation and target weighting; not a pure permutation ablation. New set has six parameterized families.')
    prior=read_json(ROOT/'protocol.json')
    if prior is not None and prior!=protocol: raise RuntimeError('Protocol mismatch')
    save_json(ROOT/'protocol.json',protocol); save_json(ROOT/'new_benchmark.json',fresh); save_json(ROOT/'old_benchmark.json',old)
    save_json(ROOT/'split.json',original_split)
    train_rows=examples(train); val_rows=examples(val)
    save_json(ROOT/'train_examples.json',train_rows); save_json(ROOT/'validation_examples.json',val_rows)
    save_json(ROOT/'teacher_source_records.json',records)
    (ROOT/'source').mkdir(exist_ok=True)
    for filename in ('run_stats_v0_4.py','stats_holdout_v2.py','stats_holdout_v1.py','run_stats_rotation_v1.py','stats_v0_3_common.py','flight_run_stats_v0_3.py','flight_run_stats_v0_1.py'):
        shutil.copy2(Path(__file__).with_name(filename),ROOT/'source'/filename)
    print('V04 FROZEN',json.dumps(protocol),flush=True)
    if torch.cuda.device_count()!=1: raise RuntimeError('One visible GPU required')
    save_json(ROOT/'environment.json',dict(gpu=torch.cuda.get_device_name(0),packages={p:importlib.metadata.version(p) for p in ('torch','transformers','peft','bitsandbytes','datasets','accelerate')}))
    token=UserSecretsClient().get_secret('HF_TOKEN')
    tokenizer=AutoTokenizer.from_pretrained(source/'adapter')
    def load_base():
        set_seed(226)
        m=AutoModelForCausalLM.from_pretrained(STUDENT,revision=BASE_REVISION,token=token,device_map={'':0},torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
        return prepare_model_for_kbit_training(m,gradient_checkpointing_kwargs={'use_reentrant':False})
    model=load_base()
    summaries={}
    for name in ('baseline','v03'):
        if name=='v03': model=PeftModel.from_pretrained(model,source/'adapter')
        summaries[name]={label:evaluate(model,tokenizer,qs,ROOT/f'{name}_{label}.json') for label,qs in (('new',fresh),('old',old))}
        print('V04 MODEL',name,json.dumps({k:v['overall'] for k,v in summaries[name].items()}),flush=True)
    del model
    import gc
    gc.collect(); torch.cuda.empty_cache()
    model=load_base()
    complete=read_json(ROOT/'training_complete.json')
    if complete:
        model=PeftModel.from_pretrained(model,ROOT/'adapter')
    else:
        model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,bias='none',task_type='CAUSAL_LM',
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
        model.config.use_cache=False
        args=TrainingArguments(output_dir=str(ROOT/'checkpoints'),max_steps=36,per_device_train_batch_size=1,
            per_device_eval_batch_size=1,gradient_accumulation_steps=8,learning_rate=5e-5,warmup_steps=2,
            lr_scheduler_type='cosine',fp16=True,logging_steps=6,eval_strategy='steps',eval_steps=12,
            save_strategy='steps',save_steps=12,save_total_limit=2,load_best_model_at_end=True,
            metric_for_best_model='eval_loss',greater_is_better=False,report_to='none',remove_unused_columns=False,
            optim='paged_adamw_8bit',seed=226)
        trainer=Trainer(model=model,args=args,train_dataset=dataset(train_rows,tokenizer),
            eval_dataset=dataset(val_rows,tokenizer),data_collator=CausalCollator(tokenizer))
        checkpoints=sorted((ROOT/'checkpoints').glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
        result=trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)
        model.save_pretrained(ROOT/'adapter'); tokenizer.save_pretrained(ROOT/'adapter')
        complete=dict(training_loss=result.training_loss,best_validation_loss=trainer.state.best_metric,
            best_checkpoint=trainer.state.best_model_checkpoint,steps=trainer.state.global_step,train_questions=len(train),
            validation_questions=len(val),train_sequences=len(train_rows),validation_sequences=len(val_rows))
        save_json(ROOT/'training_complete.json',complete); save_json(ROOT/'trainer_log.json',trainer.state.log_history)
        # Validate the saved artifact through reloading before final evaluation.
        del trainer,model
        gc.collect(); torch.cuda.empty_cache()
        model=PeftModel.from_pretrained(load_base(),ROOT/'adapter')
    summaries['v04']={label:evaluate(model,tokenizer,qs,ROOT/f'v04_{label}.json') for label,qs in (('new',fresh),('old',old))}
    summary=dict(protocol=protocol,training=complete,models=summaries,additional_teacher_calls=0,
                 adapter_sha256=hashlib.sha256((ROOT/'adapter'/'adapter_model.safetensors').read_bytes()).hexdigest())
    save_json(ROOT/'summary.json',summary); package(ROOT)
    archive=ROOT.parent/(ROOT.name+'.zip')
    print('V04 COMPLETE',json.dumps(summary),flush=True)
    print('V04 ARCHIVE',archive.stat().st_size,hashlib.sha256(archive.read_bytes()).hexdigest(),flush=True)


if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'; os.environ['TOKENIZERS_PARALLELISM']='false'
    main()
