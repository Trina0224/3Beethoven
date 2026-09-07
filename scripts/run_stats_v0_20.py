"""Two seeds, one epoch, contrast teaching; validation-only checkpoint choice."""
import os,json,gc,random,shutil,sys,zipfile,hashlib
from pathlib import Path
from stats_curriculum_v0_20 import build,digest,prompt,score
from run_stats_v0_19 import evaluate,metrics,eligible
from run_stats_v0_17 import HASHES
REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/kaggle/working/3beethoven_stats_v0_20')
DATA_SHA='e66c47dc3e45195ded1bb84a977cf8b4384dd69b6a59656b4fcfea58d63769e2'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,Trainer,TrainingArguments,set_seed
 from peft import PeftModel,prepare_model_for_kbit_training
 from kaggle_secrets import UserSecretsClient
 from torch.utils.data import SequentialSampler
 from run_stats_v0_4 import BASE_REVISION,dataset,evaluate as mc
 from flight_run_stats_v0_1 import CausalCollator
 from flight_run_stats_v0_3 import STUDENT
 seed=int(sys.argv[1]);assert seed in (2027,31415)
 ROOT.mkdir(exist_ok=True);folder=ROOT/f'seed_{seed}';folder.mkdir(exist_ok=True)
 data=read(REPO/'docs/STATS_V0_20_FROZEN_QUESTIONS.json');assert digest(data)==DATA_SHA and data==build()
 # Only a matching saved original parent is allowed, never a replicated adapter.
 parents=[p.parent for p in Path(os.environ.get('V20_PARENT_ROOT','/kaggle/input')).rglob('adapter_model.safetensors') if p.parent.parent.name=='3beethoven_stats_v0_15' and sha(p)==HASHES[15]]
 assert parents,'Mount saved Version 42 output containing original v15'
 parent=parents[0];token=UserSecretsClient().get_secret('HF_TOKEN');tok=AutoTokenizer.from_pretrained(parent)
 def load(path,training=False):
  set_seed(seed)
  m=AutoModelForCausalLM.from_pretrained(STUDENT,revision=BASE_REVISION,token=token,device_map={'':0},torch_dtype=torch.float16,quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
  if training:m=prepare_model_for_kbit_training(m,gradient_checkpointing_kwargs={'use_reentrant':False})
  m=PeftModel.from_pretrained(m,path,is_trainable=training);m.config.use_cache=not training;return m
 old=read(REPO/'docs/STATS_V0_19_FROZEN_QUESTIONS.json');anchor=read(REPO/'docs/STATS_PERMANENT_ANCHOR_V1.json')
 save(folder/'manifest.json',dict(seed=seed,data_sha256=DATA_SHA,parent_sha256=HASHES[15],base_revision=BASE_REVISION,teacher_calls=0,max_steps=48,gpu=torch.cuda.get_device_name(0)))
 # Parent validation/test is cached across seeds: greedy fixed-weight inference.
 baseline_dir=ROOT/'parent';baseline_dir.mkdir(exist_ok=True)
 if not (baseline_dir/'complete.json').exists():
  model=load(parent);bm={}
  for name,qs in [('old_validation',old['old_validation']),('event_validation',data['validation']),('event_test',data['test']),('old_test',old['old_test']),('new_test',old['new_test'])]:
   bm[name]=evaluate(model,tok,qs,baseline_dir/f'{name}.json');print('PARENT',name,bm[name],flush=True)
  mc(model,tok,anchor['mc']['questions'],baseline_dir/'mc.json');save(baseline_dir/'complete.json',bm)
  del model;gc.collect();torch.cuda.empty_cache()
 baseline=read(baseline_dir/'complete.json')
 if not (folder/'training_complete.json').exists():
  rng=random.Random(seed);groups=[data['train'][i:i+4] for i in range(0,192,4)];rng.shuffle(groups);replay=list(data['replay']);rng.shuffle(replay)
  rows=[]
  for i,group in enumerate(groups):
   rows.extend(dict(prompt=prompt(q),target=q['target'],source_id=q['id']) for q in group);rows.extend(replay[i*4:i*4+4])
  save(folder/'training_order.json',rows)
  model=load(parent,True)
  class OrderedTrainer(Trainer):
   def _get_train_sampler(self,train_dataset=None):return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)
  args=TrainingArguments(output_dir=str(folder/'checkpoints'),num_train_epochs=1,max_steps=48,per_device_train_batch_size=1,gradient_accumulation_steps=8,learning_rate=2e-5,lr_scheduler_type='constant',warmup_steps=0,fp16=True,logging_steps=8,save_strategy='steps',save_steps=24,report_to='none',remove_unused_columns=False,optim='paged_adamw_8bit',seed=seed,disable_tqdm=True)
  trainer=OrderedTrainer(model=model,args=args,train_dataset=dataset(rows,tok),data_collator=CausalCollator(tok))
  checkpoints=sorted((folder/'checkpoints').glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
  result=trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)
  assert trainer.state.global_step==48
  for step in (24,48):tok.save_pretrained(folder/'checkpoints'/f'checkpoint-{step}')
  save(folder/'training_complete.json',dict(steps=48,loss=result.training_loss,logs=trainer.state.log_history))
  del trainer,model;gc.collect();torch.cuda.empty_cache()
 candidates=[]
 for step in (24,48):
  checkpoint=folder/'checkpoints'/f'checkpoint-{step}';out=folder/f'step_{step}';out.mkdir(exist_ok=True)
  model=load(checkpoint)
  val={name:evaluate(model,tok,qs,out/f'{name}.json') for name,qs in [('old_validation',old['old_validation']),('event_validation',data['validation'])]}
  ok=eligible(val['old_validation'],baseline['old_validation']);candidates.append(dict(step=step,eligible=ok,metrics=val))
  # Keep portable adapters independent from optimizer checkpoints.
  model.save_pretrained(out/'adapter');tok.save_pretrained(out/'adapter')
  del model;gc.collect();torch.cuda.empty_cache();print('VALIDATION',seed,step,val,ok,flush=True)
 allowed=[r for r in candidates if r['eligible']];chosen=max(allowed,key=lambda r:(r['metrics']['event_validation']['correct'],-r['step'])) if allowed else candidates[-1]
 selection=dict(candidates=candidates,selected_step=chosen['step'] if allowed else None,diagnostic_step=None if allowed else 48)
 save(folder/'selection.json',selection)
 model=load(folder/f"step_{chosen['step']}"/'adapter');tests={}
 for name,qs in [('event_test',data['test']),('old_test',old['old_test']),('new_test',old['new_test'])]:
  tests[name]=evaluate(model,tok,qs,folder/f'{name}.json');print('TEST',seed,name,tests[name],flush=True)
 mc(model,tok,anchor['mc']['questions'],folder/'mc.json')
 import safetensors.torch
 weights=safetensors.torch.load_file(str(folder/f"step_{chosen['step']}"/'adapter/adapter_model.safetensors'))
 assert all(torch.isfinite(t).all() for t in weights.values())
 tests['mc']=dict(correct=sum(r['correct'] for r in read(folder/'mc.json')),n=240)
 save(folder/'complete.json',dict(selection=selection,tests=tests,finite_tensors=len(weights),adapter_sha256=sha(folder/f"step_{chosen['step']}"/'adapter/adapter_model.safetensors')))
 del model;gc.collect();torch.cuda.empty_cache()
 with zipfile.ZipFile(ROOT/f'seed_{seed}.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in folder.rglob('*'):
   if p.is_file() and 'checkpoints' not in p.parts:z.write(p,p.relative_to(ROOT))
 print('SEED COMPLETE',seed,tests,flush=True)
if __name__=='__main__':
 os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['TOKENIZERS_PARALLELISM']='false';main()
