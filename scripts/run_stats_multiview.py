"""Preregistered two-seed multiview response-distillation pilot.

Run baseline, both train invocations, offline review/selection, then probes.
No pause or early stopping based on validation; no automatic numeric-only credit.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse, hashlib, importlib.metadata, json, math, random
from collections import defaultdict
from pathlib import Path
from flight_run_stats_v0_3 import STUDENT, read_json as read, save_json as save
from run_stats_v0_4 import BASE_REVISION, dataset
from run_stats_v0_17 import HASHES
from stats_consolidation_pilot import digest, build
from stats_consolidation_grader import grader_fingerprint
from stats_consolidation_eval import evaluate
from run_stats_consolidation_compare import (suite_metrics, gate_result, locate_v15,
    token_accounting, validation_suites, validate_verified_rows)
from stats_baseline_contract import NUMERICAL_PREPARATION, baseline_manifest, validate_baseline_manifest

REPO=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def ordered_rows(rows,seed):
    groups=defaultdict(list)
    for row in rows:groups[row['source_group']+':'+row['story_id']].append(row)
    keys=sorted(groups);random.Random(seed).shuffle(keys)
    return [row for key in keys for row in groups[key]]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('baseline','train','probe'),required=True)
    p.add_argument('--seed',type=int,choices=(2027,31415))
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    decision=read(REPO/'docs/STATS_MULTIVIEW_DECISION.json')
    corpus=read(REPO/'docs/STATS_MULTIVIEW_CORPUS.json')
    assert decision['status']=='frozen_for_execution'
    assert digest(corpus['train'])==decision['train_sha256']
    assert digest(corpus['transfer_probe'])==decision['probe_sha256']
    assert len(corpus['train'])==516 and math.ceil(len(corpus['train'])/8)==65
    assert decision['validation_gate']==read(REPO/'docs/STATS_CONSOLIDATION_DECISION_DRAFT.json')['validation_gate']
    validate_verified_rows(read(a.input/'3beethoven_first_students/teacher/verified_teacher_rows.json'),build()[0])
    parent=locate_v15(a.input/'first_batch_parent',HASHES[15])
    import torch
    from kaggle_secrets import UserSecretsClient
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,Trainer,TrainerCallback,TrainingArguments,set_seed
    from torch.utils.data import SequentialSampler
    from flight_run_stats_v0_1 import CausalCollator
    assert torch.cuda.device_count()==1
    env={k:importlib.metadata.version(k) for k in ('torch','transformers','peft','bitsandbytes','accelerate','datasets')}
    assert env==decision['environment'],env
    save(a.root/'environment.json',env)
    set_seed(a.seed or 2027)
    token=UserSecretsClient().get_secret('HF_TOKEN')
    tokenizer=AutoTokenizer.from_pretrained(STUDENT,revision=BASE_REVISION,token=token)
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    context={'v15_baseline_adapter_sha256':HASHES[15],'base_revision':BASE_REVISION,
        'selection_matrix_sha256':sha(REPO/'docs/STATS_CONSOLIDATION_SELECTION_VALIDATION.json'),
        'grader_fingerprint':grader_fingerprint(),'environment':env}
    suites=validation_suites()
    reviews={(r['question_sha256'],r['raw_sha256']):r for r in read(REPO/'docs/STATS_MULTIVIEW_PRIOR_REVIEWS.json')}
    def reviewed(model,questions,path):
        rows=evaluate(model,tokenizer,questions,path)
        automatic=Path(path).with_name(Path(path).stem+'_automatic.json')
        save(automatic,rows)
        for row in rows:
            prior=reviews.get((row['question_sha256'],row['raw_sha256']))
            if row.get('review_required') and prior:
                for key in ('math_correct','review_required','primary_correct','correct','reason'):row[key]=prior[key]
                row['semantic_review_reused']='Mounted Version54 identical question and raw hashes'
        save(path,rows);return rows
    adapter=parent
    if a.mode=='probe':
        assert all(read(a.root/f'seed_{seed}/training_complete.json') for seed in (2027,31415))
        if a.seed:
            selection=read(a.root/f'seed_{a.seed}/selection.json')
            assert selection and selection['all_pending_resolved']
            adapter=a.root/f'seed_{a.seed}/step_{selection["fixed_step"]}/adapter'
    model=AutoModelForCausalLM.from_pretrained(STUDENT,revision=BASE_REVISION,token=token,
        device_map={'':0},torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
    model=prepare_model_for_kbit_training(model,gradient_checkpointing_kwargs={'use_reentrant':False})
    model=PeftModel.from_pretrained(model,adapter,is_trainable=True)
    model.config.use_cache=False
    if a.mode=='baseline':
        metrics={name:suite_metrics(reviewed(model,questions,a.root/'v15_selection_rows'/f'{name}.json')) for name,questions in suites.items()}
        assert not sum(m['pending'] for m in metrics.values()),'Review baseline before training'
        save(a.root/'v15_selection_baseline.json',metrics)
        save(a.root/'v15_selection_baseline_contract.json',baseline_manifest(metrics,context))
        print('MULTIVIEW_BASELINE_COMPLETE',json.dumps(metrics),flush=True);return
    if a.mode=='probe':
        folder=a.root/'probe'/('v15' if a.seed is None else str(a.seed))
        rows=reviewed(model,corpus['transfer_probe'],folder/'rows.json')
        save(folder/'contract.json',{'adapter_sha256':sha(adapter/'adapter_model.safetensors'),
            'probe_sha256':decision['probe_sha256'],'numerical_preparation':NUMERICAL_PREPARATION})
        print('MULTIVIEW_PROBE_COMPLETE',json.dumps({'seed':a.seed,**suite_metrics(rows)}),flush=True);return
    assert a.seed is not None
    baseline=read(a.root/'v15_selection_baseline.json')
    validate_baseline_manifest(baseline,read(a.root/'v15_selection_baseline_contract.json'),context)
    output=a.root/f'seed_{a.seed}'
    assert not output.exists(),'Fresh uninterrupted invocation required; investigate interruption separately'
    output.mkdir()
    rows=ordered_rows(corpus['train'],a.seed);prepared=dataset(rows,tokenizer)
    save(output/'training_order.json',rows)
    save(output/'contract.json',{'decision_sha256':digest(decision),'corpus_sha256':digest(corpus),
        'parent_sha256':sha(parent/'adapter_model.safetensors'),'base_revision':BASE_REVISION,
        'seed':a.seed,'training_order_sha256':digest(rows),'token_accounting':token_accounting(rows,tokenizer),
        'numerical_preparation':NUMERICAL_PREPARATION,'environment':env,'grader_fingerprint':grader_fingerprint(),
        'runner_sha256':sha(__file__)})
    class Validation(TrainerCallback):
        def on_step_end(self,args,state,control,model=None,**kwargs):
            if state.global_step in decision['validation_steps']:
                folder=output/f'step_{state.global_step}'
                metrics={name:suite_metrics(reviewed(model,questions,folder/f'{name}.json')) for name,questions in suites.items()}
                history=read(output/'validation_history.json',[])
                history.append({'step':state.global_step,'metrics':metrics,'provisional_gate':gate_result(metrics,baseline,decision['validation_gate'])})
                save(output/'validation_history.json',history)
                model.save_pretrained(folder/'adapter');tokenizer.save_pretrained(folder/'adapter')
                print('MULTIVIEW_CHECKPOINT',json.dumps({'seed':a.seed,'step':state.global_step,
                    'scores':{k:v['correct'] for k,v in metrics.items()},'pending':sum(v['pending'] for v in metrics.values())}),flush=True)
                control.should_save=True
            return control
    class OrderedTrainer(Trainer):
        def _get_train_sampler(self,train_dataset=None):return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)
        def training_step(self,model,inputs,*args,**kwargs):
            item={'global_step_before':self.state.global_step,
                'input_ids_sha256':digest(inputs['input_ids'].detach().cpu().tolist()),
                'labels_sha256':digest(inputs['labels'].detach().cpu().tolist()),
                'supervised_tokens':int((inputs['labels']!=-100).sum().item())}
            loss=super().training_step(model,inputs,*args,**kwargs)
            with (output/'microbatches.jsonl').open('a') as stream:stream.write(json.dumps(item)+'\n')
            return loss
    train_args=TrainingArguments(output_dir=str(output/'checkpoints'),num_train_epochs=1,max_steps=-1,
        per_device_train_batch_size=1,gradient_accumulation_steps=8,learning_rate=5e-6,
        lr_scheduler_type='constant',warmup_steps=0,fp16=True,logging_steps=12,
        save_strategy='no',save_total_limit=4,report_to='none',remove_unused_columns=False,
        optim='paged_adamw_8bit',seed=a.seed,data_seed=a.seed,disable_tqdm=True)
    trainer=OrderedTrainer(model=model,args=train_args,train_dataset=prepared,
        data_collator=CausalCollator(tokenizer),callbacks=[Validation()])
    result=trainer.train()
    trace=[json.loads(line) for line in (output/'microbatches.jsonl').read_text().splitlines()]
    assert len(trace)==len(prepared)==516
    for i,(actual,expected) in enumerate(zip(trace,prepared)):
        assert actual['input_ids_sha256']==digest([expected['input_ids']])
        assert actual['labels_sha256']==digest([expected['labels']])
        assert actual['global_step_before']==i//8
    assert trainer.state.global_step==65
    save(output/'training_complete.json',{'global_steps':65,'train_loss':result.training_loss,
        'uninterrupted':True,'actual_microbatches_verified':516,
        'actual_supervised_tokens':sum(r['supervised_tokens'] for r in trace),
        'selection':'awaits post-training semantic review; probe not inspected'})
    print('MULTIVIEW_TRAIN_COMPLETE',a.seed,flush=True)

if __name__=='__main__':main()
