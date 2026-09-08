"""One fixed-final student; direct-interface paired evaluation, no automatic retry."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse,json,hashlib,importlib.metadata
from pathlib import Path
from stats_final_clear import digest
from stats_consolidation_eval import evaluate
from run_stats_v0_4 import dataset,BASE_REVISION
from flight_run_stats_v0_1 import CausalCollator
BASE='meta-llama/Llama-3.2-3B-Instruct'
PARENT='9369d52de4a886df9da0c872cd41bd4e01af0a38bf02ad724b5951c1a6b9f5d3'
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['baseline','train','parent_test','student'],required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
 d=json.loads((ROOT/'docs/STATS_FINAL_CLEAR_DATA.json').read_text());m=d['manifest'];assert m['status']=='frozen_before_inference'
 for s in ('train','development','test'):assert digest(d[s])==m['split_sha256'][s]
 assert sha(a.parent/'adapter_model.safetensors')==PARENT
 import torch
 from torch.utils.data import SequentialSampler
 from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,Trainer,TrainingArguments,set_seed
 from peft import PeftModel,prepare_model_for_kbit_training
 from kaggle_secrets import UserSecretsClient
 set_seed(2027)
 env={k:importlib.metadata.version(k) for k in ('torch','transformers','peft','bitsandbytes','accelerate','datasets')}
 assert env=={'torch':'2.10.0+cu128','transformers':'5.0.0','peft':'0.19.1','bitsandbytes':'0.50.2','accelerate':'1.13.0','datasets':'5.0.0'},env
 token=UserSecretsClient().get_secret('HF_TOKEN')
 tok=AutoTokenizer.from_pretrained(BASE,revision=BASE_REVISION,token=token)
 if tok.pad_token is None:tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(BASE,revision=BASE_REVISION,token=token,device_map={'':0},torch_dtype=torch.float16,quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
 model=prepare_model_for_kbit_training(model,gradient_checkpointing_kwargs={'use_reentrant':False})
 adapter=a.root/'student/adapter' if a.mode=='student' else a.parent
 if a.mode=='student':
  completed=json.loads((a.root/'student/training_complete.json').read_text());assert sha(adapter/'adapter_model.safetensors')==completed['adapter_sha256']
 model=PeftModel.from_pretrained(model,adapter,is_trainable=a.mode=='train');model.config.use_cache=False
 contract=dict(mode=a.mode,environment=env,parent_sha256=PARENT,loaded_adapter_sha256=sha(adapter/'adapter_model.safetensors'),base=BASE,base_revision=BASE_REVISION,split_sha256=m['split_sha256'])
 save(a.root/(a.mode+'_contract.json'),contract)
 if a.mode!='train':
  suites={'baseline':['development'],'parent_test':['test'],'student':['development','test']}[a.mode];name='student' if a.mode=='student' else 'v15'
  for suite in suites:
   rr=evaluate(model,tok,d[suite],a.root/'evaluation'/name/(suite+'.json'))
   print('FINAL_CLEAR_EVAL',name,suite,json.dumps({'n':len(rr),'correct':sum(x['primary_correct'] for x in rr),'pending':sum(x['review_required'] for x in rr)}),flush=True)
  return
 folder=a.root/'student';assert not folder.exists(),'Refuse a second training invocation';folder.mkdir()
 prepared=dataset(d['train'],tok);assert len(prepared)==1152
 class AuditTrainer(Trainer):
  def _get_train_sampler(self,train_dataset=None):return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)
  def training_step(self,model,inputs,*args,**kwargs):
   step=self.state.global_step;item=dict(step=step,input_sha256=digest(inputs['input_ids'].detach().cpu().tolist()),labels_sha256=digest(inputs['labels'].detach().cpu().tolist()),supervised_tokens=int((inputs['labels']!=-100).sum()))
   loss=super().training_step(model,inputs,*args,**kwargs)
   with (folder/'microbatches.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
   return loss
 args=TrainingArguments(output_dir=str(folder/'checkpoints'),num_train_epochs=1,per_device_train_batch_size=1,gradient_accumulation_steps=8,learning_rate=2e-5,lr_scheduler_type='constant',warmup_steps=0,fp16=True,logging_steps=12,save_strategy='steps',save_steps=72,save_total_limit=1,report_to='none',remove_unused_columns=False,optim='paged_adamw_8bit',seed=2027,data_seed=2027,disable_tqdm=True)
 tr=AuditTrainer(model=model,args=args,train_dataset=prepared,data_collator=CausalCollator(tok));res=tr.train();assert tr.state.global_step==144
 trace=[json.loads(l) for l in (folder/'microbatches.jsonl').read_text().splitlines()];assert len(trace)==1152
 for i,(x,y) in enumerate(zip(trace,prepared)):
  assert x['step']==i//8 and x['input_sha256']==digest([y['input_ids']]) and x['labels_sha256']==digest([y['labels']])
 model.save_pretrained(folder/'adapter');tok.save_pretrained(folder/'adapter')
 from safetensors.torch import load_file
 tensors=load_file(folder/'adapter/adapter_model.safetensors');assert all(torch.isfinite(t).all() for t in tensors.values())
 save(folder/'training_complete.json',dict(updates=144,rows=1152,loss=res.training_loss,supervised_tokens=sum(r['supervised_tokens'] for r in trace),finite_tensors=len(tensors),adapter_sha256=sha(folder/'adapter/adapter_model.safetensors'),selection='fixed_final',log_history=tr.state.log_history))
 print('FINAL_CLEAR_TRAIN_COMPLETE',flush=True)
if __name__=='__main__':main()
