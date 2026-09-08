"""Two fixed-final students, matched example counts, unchanged historical grader."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from flight_run_stats_v0_3 import STUDENT, read_json as read, save_json as save
from run_stats_v0_4 import BASE_REVISION, dataset
from run_stats_v0_17 import HASHES
from stats_consolidation_pilot import digest
from stats_consolidation_grader import grader_fingerprint
from stats_consolidation_eval import evaluate
from run_stats_consolidation_compare import suite_metrics, token_accounting, validation_suites
from stats_baseline_contract import NUMERICAL_PREPARATION

ROOT=Path(__file__).resolve().parents[1]
PRIOR='92eca7f1257880cb5c69bccd7bda8d600f0268bea62c9622d97551b1fbf6506a'
ARMS=('semantic_course','repeat_control')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('train','eval'),required=True)
    p.add_argument('--arm',choices=(*ARMS,'v15','prior_v55'),required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--resume',type=Path)
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    data=read(ROOT/'docs/STATS_SEMANTIC_COURSE_DATA.json');manifest=data['manifest']
    assert manifest['status']=='frozen_before_training'
    for arm in ARMS:
        assert len(data['arms'][arm])==1006 and digest(data['arms'][arm])==manifest['arm_sha256'][arm]
    assert len(data['holdout'])==128 and digest(data['holdout'])==manifest['holdout_sha256']
    assert grader_fingerprint()=='fb3599dbc63ba1aa99aef4aa9c7910fc0c049246acb14875a891df59ffc13e03'
    import torch
    from kaggle_secrets import UserSecretsClient
    from peft import PeftModel,prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,Trainer,TrainingArguments,set_seed
    from torch.utils.data import SequentialSampler
    from flight_run_stats_v0_1 import CausalCollator
    env={k:importlib.metadata.version(k) for k in ('torch','transformers','peft','bitsandbytes','accelerate','datasets')}
    expected={'torch':'2.10.0+cu128','transformers':'5.0.0','peft':'0.19.1','bitsandbytes':'0.50.2','accelerate':'1.13.0','datasets':'5.0.0'}
    assert env==expected,env
    set_seed(2027);assert torch.cuda.device_count()==1
    parent=a.input/'portable/v15_parent';prior=a.input/'portable/prior_v55_2027'
    assert sha(parent/'adapter_model.safetensors')==HASHES[15]
    assert sha(prior/'adapter_model.safetensors')==PRIOR
    adapter=parent
    if a.mode=='eval':
        assert all(read(a.root/arm/'training_complete.json') for arm in ARMS)
        if a.arm in ARMS:
            adapter=a.root/a.arm/'adapter'
            assert sha(adapter/'adapter_model.safetensors')==read(a.root/a.arm/'training_complete.json')['adapter_sha256']
        elif a.arm=='prior_v55':adapter=prior
    else:assert a.arm in ARMS
    token=UserSecretsClient().get_secret('HF_TOKEN')
    tokenizer=AutoTokenizer.from_pretrained(STUDENT,revision=BASE_REVISION,token=token)
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(STUDENT,revision=BASE_REVISION,token=token,
        device_map={'':0},torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
    model=prepare_model_for_kbit_training(model,gradient_checkpointing_kwargs={'use_reentrant':False})
    model=PeftModel.from_pretrained(model,adapter,is_trainable=True);model.config.use_cache=False
    context=dict(arm=a.arm,parent_sha256=HASHES[15],base_revision=BASE_REVISION,environment=env,
        numerical_preparation=NUMERICAL_PREPARATION,grader_fingerprint=grader_fingerprint(),
        runner_sha256=sha(__file__),holdout_sha256=manifest['holdout_sha256'])
    if a.mode=='eval':
        folder=a.root/'evaluation'/a.arm;folder.mkdir(parents=True,exist_ok=True)
        suites=dict(validation_suites(),fresh=data['holdout'])
        for name,questions in suites.items():
            rows=evaluate(model,tokenizer,questions,folder/f'{name}.json')
            save(folder/f'{name}_automatic.json',rows)
            print('COURSE_EVAL_SUITE',json.dumps(dict(arm=a.arm,suite=name,**suite_metrics(rows))),flush=True)
        save(folder/'contract.json',dict(context,adapter_sha256=sha(adapter/'adapter_model.safetensors')))
        print('COURSE_EVAL_COMPLETE',a.arm,flush=True);return
    folder=a.root/a.arm
    if not a.resume:assert not folder.exists(),'Existing training output; inspect rather than silently restart'
    folder.mkdir(parents=True,exist_ok=True)
    rows=data['arms'][a.arm];prepared=dataset(rows,tokenizer)
    contract=dict(context,training_sha256=digest(rows),seed=2027,learning_rate=5e-6,
                  epochs=1,updates=126,token_accounting=token_accounting(rows,tokenizer))
    old=read(folder/'contract.json');assert old is None or old==contract
    save(folder/'contract.json',contract)
    class OrderedTrainer(Trainer):
        def _get_train_sampler(self,train_dataset=None):
            return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)
        def training_step(self,model,inputs,*args,**kwargs):
            item=dict(global_step_before=self.state.global_step,
                input_ids_sha256=digest(inputs['input_ids'].detach().cpu().tolist()),
                labels_sha256=digest(inputs['labels'].detach().cpu().tolist()),
                supervised_tokens=int((inputs['labels']!=-100).sum().item()))
            loss=super().training_step(model,inputs,*args,**kwargs)
            with (folder/'microbatches.jsonl').open('a') as stream:stream.write(json.dumps(item)+'\n')
            return loss
    args=TrainingArguments(output_dir=str(folder/'checkpoints'),num_train_epochs=1,max_steps=-1,
        per_device_train_batch_size=1,gradient_accumulation_steps=8,learning_rate=5e-6,
        lr_scheduler_type='constant',warmup_steps=0,fp16=True,logging_steps=12,
        save_strategy='steps',save_steps=63,save_total_limit=1,report_to='none',
        remove_unused_columns=False,optim='paged_adamw_8bit',seed=2027,data_seed=2027,disable_tqdm=True)
    trainer=OrderedTrainer(model=model,args=args,train_dataset=prepared,data_collator=CausalCollator(tokenizer))
    if a.resume:
        state=read(a.resume/'trainer_state.json');step=state['global_step']
        assert step==63 and a.resume.resolve().is_relative_to((folder/'checkpoints').resolve())
        trace_path=folder/'microbatches.jsonl'
        trace=trace_path.read_text().splitlines();save(folder/'interrupted_trace.json',trace)
        trace_path.write_text('\n'.join(trace[:step*8])+'\n')
    result=trainer.train(resume_from_checkpoint=str(a.resume) if a.resume else None)
    assert trainer.state.global_step==126
    model.save_pretrained(folder/'adapter');tokenizer.save_pretrained(folder/'adapter')
    trace=[json.loads(line) for line in (folder/'microbatches.jsonl').read_text().splitlines()]
    assert len(trace)==len(prepared)==1006
    for i,(actual,expected) in enumerate(zip(trace,prepared)):
        assert actual['global_step_before']==i//8
        assert actual['input_ids_sha256']==digest([expected['input_ids']])
        assert actual['labels_sha256']==digest([expected['labels']])
    save(folder/'training_complete.json',dict(global_steps=126,train_loss=result.training_loss,
        verified_microbatches=len(trace),supervised_tokens=sum(r['supervised_tokens'] for r in trace),
        adapter_sha256=sha(folder/'adapter/adapter_model.safetensors'),fixed_selection='final_step_126',
        resumed_from=str(a.resume) if a.resume else None))
    print('COURSE_TRAIN_COMPLETE',a.arm,flush=True)

if __name__=='__main__':main()
