"""v0.14 -> rule-scaffolded decomposed response distillation. Run on Kaggle, not at import time."""
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
from pathlib import Path
from stats_curriculum_v0_15 import build, digest, score, prompt
from prepare_stats_v0_15 import teacher_rows

ROOT=Path('/kaggle/working/3beethoven_stats_v0_15')
V14=Path('/kaggle/working/3beethoven_stats_v0_14')
SOURCE_SHA='c7def77757fefaaf41db6938500159795a47503dac54d72d79113de47a3239a5'
DATA_SHA='d7df943fa318ffbaeffabd12834a8f345c54f71c48f8c8cc1cbef8c78ba55f1f'


def save(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n')
    tmp.replace(path)


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def restore():
    target=V14/'adapter/adapter_model.safetensors'
    if not target.exists():
        import kagglehub
        saved=Path(kagglehub.notebook_output_download('trinashih/3beethoven-v0-2/versions/28'))
        candidates=sorted((p for p in saved.rglob(V14.name) if (p/'adapter/adapter_model.safetensors').exists()),key=lambda p:len(p.parts))
        if not candidates: raise RuntimeError('Saved v0.14 adapter not found; do not train from another checkpoint.')
        shutil.copytree(candidates[0],V14,dirs_exist_ok=True)
    if hashlib.sha256(target.read_bytes()).hexdigest()!=SOURCE_SHA:
        raise RuntimeError('v0.14 adapter hash mismatch')


def evaluate_formulations(model,tok,questions,path):
    import torch
    rows=read(path,[])
    expected={q['id']:q for q in questions}
    assert len({r['id'] for r in rows})==len(rows)
    for r in rows:
        assert r['question_sha256']==digest(expected[r['id']])
        assert r['prompt']==prompt(expected[r['id']])
    done={r['id'] for r in rows}
    model.eval();model.config.use_cache=True
    for q in questions:
        if q['id'] in done: continue
        p=prompt(q)
        chat=tok.apply_chat_template([dict(role='user',content=p)],tokenize=False,add_generation_prompt=True)
        inputs=tok(chat,add_special_tokens=False,return_tensors='pt').to(model.device)
        with torch.inference_mode():
            output=model.generate(**inputs,max_new_tokens=160,do_sample=False,pad_token_id=tok.eos_token_id)[0][inputs['input_ids'].shape[-1]:]
        raw=tok.decode(output,skip_special_tokens=True).strip()
        rows.append(dict(id=q['id'],category=q['category'],question_sha256=digest(q),prompt=p,
                         raw=raw,generated_tokens=len(output),hit_token_limit=len(output)==160,**score(raw,q)))
        save(path,rows)
        if len(rows)%12==0: print('V15 EVAL',path.stem,len(rows),'/',len(questions),flush=True)
    def counts(subset):
        return dict(n=len(subset),**{k:sum(r[k] for r in subset) for k in
                    ('correct','executable','review_required','invalid')})
    return dict(overall=counts(rows),by_category={c:counts([r for r in rows if r['category']==c]) for c in sorted({q['category'] for q in questions})})


def main():
    import torch
    from kaggle_secrets import UserSecretsClient
    from peft import PeftModel,prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,Trainer,TrainingArguments,set_seed
    from flight_run_stats_v0_3 import STUDENT,package
    from flight_run_stats_v0_1 import CausalCollator
    from run_stats_v0_4 import BASE_REVISION,dataset,evaluate
    from stats_holdout_v1 import questions as old_questions

    repo=Path(__file__).resolve().parents[1]
    data=build()
    assert digest(data)==DATA_SHA
    assert data==read(repo/'docs/STATS_V0_15_FROZEN_QUESTIONS.json')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    gate=read(repo/'docs/STATS_V0_14_TEACHER_SCAFFOLD_RESULTS.json')
    if not gate or not gate['summary']['passed']: raise RuntimeError('New rule-scaffolded teacher gate has not passed')
    restore()
    ROOT.mkdir(exist_ok=True)
    protocol=dict(data_sha256=DATA_SHA,source_adapter_sha256=SOURCE_SHA,source='v0.14',
        base_revision=BASE_REVISION,method='Rule-scaffolded 70B response distillation; decomposed moment questions and weak-family practice; 32 replay examples',
        teacher_check='New rule-scaffolded diagnostic; earlier independent diagnostic remains failed',
        teacher_check_sha256=gate['summary']['protocol_sha256'],
        training_teacher_reminders=True,
        seed=1515,epochs=3,lr=2e-5,effective_batch=8,new_teacher_calls=0,
        checkpoint_selection='validation loss only',max_new_tokens=160,
        auto_score='Frozen formulation-v2: previously reviewed equivalences; no test-time grader changes',
        test_role='Frozen v15 wording/parameter holdout; no teacher test targets; not general reasoning benchmark')
    previous=read(ROOT/'training_protocol.json')
    assert previous is None or previous==protocol
    save(ROOT/'training_protocol.json',protocol)
    save(ROOT/'frozen_questions.json',data)
    source=ROOT/'source';source.mkdir(exist_ok=True)
    for file in (repo/'scripts').glob('*.py'): shutil.copy2(file,source/file.name)
    rows=teacher_rows(repo)
    assert len(rows['train'])>=160 and len(rows['validation'])>=24
    for s in rows: save(ROOT/f'{s}_examples.json',rows[s])
    token=UserSecretsClient().get_secret('HF_TOKEN')
    tok=AutoTokenizer.from_pretrained(V14/'adapter')
    lengths=[len(tok.apply_chat_template([dict(role='user',content=r['prompt']),dict(role='assistant',content=r['target'])],tokenize=True,return_dict=False)) for subset in rows.values() for r in subset]
    assert max(lengths)<=512
    save(ROOT/'environment.json',dict(gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        max_sequence_tokens=max(lengths),packages={p:importlib.metadata.version(p) for p in ('torch','transformers','peft','bitsandbytes','datasets','accelerate')}))

    def base():
        set_seed(1515)
        m=AutoModelForCausalLM.from_pretrained(STUDENT,revision=BASE_REVISION,token=token,
            device_map={'':0},torch_dtype=torch.float16,quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
        return prepare_model_for_kbit_training(m,gradient_checkpointing_kwargs={'use_reentrant':False})

    training=read(ROOT/'training_complete.json')
    if not training:
        model=PeftModel.from_pretrained(base(),V14/'adapter',is_trainable=True)
        model.config.use_cache=False
        args=TrainingArguments(output_dir=str(ROOT/'checkpoints'),num_train_epochs=3,
            per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=8,
            learning_rate=2e-5,warmup_steps=5,lr_scheduler_type='cosine',fp16=True,
            logging_steps=8,eval_strategy='epoch',save_strategy='epoch',save_total_limit=2,
            load_best_model_at_end=True,metric_for_best_model='eval_loss',greater_is_better=False,
            report_to='none',remove_unused_columns=False,optim='paged_adamw_8bit',seed=1515)
        trainer=Trainer(model=model,args=args,train_dataset=dataset(rows['train'],tok),
                        eval_dataset=dataset(rows['validation'],tok),data_collator=CausalCollator(tok))
        checkpoints=sorted((ROOT/'checkpoints').glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
        result=trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)
        model.save_pretrained(ROOT/'adapter');tok.save_pretrained(ROOT/'adapter')
        training=dict(steps=trainer.state.global_step,best_checkpoint=trainer.state.best_model_checkpoint,
            best_validation_loss=trainer.state.best_metric,training_loss=result.training_loss)
        save(ROOT/'training_complete.json',training);save(ROOT/'trainer_log.json',trainer.state.log_history)
        package(ROOT)
        print('V15 TRAINED',json.dumps(training),flush=True)
        del trainer,model;gc.collect();torch.cuda.empty_cache()
    results={}
    for name,adapter in (('baseline',None),('v14',V14/'adapter'),('v15',ROOT/'adapter')):
        model=base()
        if adapter: model=PeftModel.from_pretrained(model,adapter)
        results[name]=dict(formulation=evaluate_formulations(model,tok,data['test'],ROOT/f'{name}_formulation.json'),
                           retention=evaluate(model,tok,old_questions(),ROOT/f'{name}_old.json'))
        save(ROOT/f'{name}_metrics.json',results[name]);package(ROOT)
        print('V15 MODEL',name,json.dumps(results[name]),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    summary=dict(protocol=protocol,training=training,models=results,new_teacher_calls=0,
        adapter_sha256=hashlib.sha256((ROOT/'adapter/adapter_model.safetensors').read_bytes()).hexdigest(),
        review='Raw-output review required before substantive conclusions; no success claim from auto-score alone')
    save(ROOT/'summary.json',summary);package(ROOT)
    archive=ROOT.with_suffix('.zip')
    print('V15 COMPLETE',json.dumps(summary),flush=True)
    print('V15 ARCHIVE',archive.stat().st_size,hashlib.sha256(archive.read_bytes()).hexdigest(),flush=True)


if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    main()
