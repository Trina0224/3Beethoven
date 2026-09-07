"""Same inference loading for v15 seed MC; preserve original runner outputs."""
import os,json,gc,shutil,hashlib,sys
from pathlib import Path
INPUT=Path('/kaggle/input/notebooks/trinashih/3beethoven-v0-2')
ROOT=Path('/kaggle/working/seed_mc_harmonized')
def main():
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,set_seed
 from peft import PeftModel
 from kaggle_secrets import UserSecretsClient
 from flight_run_stats_v0_3 import STUDENT,save_json as save
 from run_stats_v0_4 import BASE_REVISION,evaluate
 from stats_v0_3_common import parse_answer
 source=INPUT/'3beethoven_seed_replication';ROOT.mkdir(exist_ok=True)
 all_results=json.loads((source/'all_results.json').read_text())
 receipt=INPUT/'seed_replication_backup_receipt.json'
 assert receipt.is_file(), 'Training backup receipt is required'
 shutil.copy2(receipt,ROOT/receipt.name)
 anchor=json.loads((INPUT/'seed_repo/docs/STATS_PERMANENT_ANCHOR_V1.json').read_text())
 expected={(r['id'],r['shift']):r['expected'] for r in anchor['mc']['rotations']}
 token=UserSecretsClient().get_secret('HF_TOKEN');summary={}
 for seed in (1919,2027,31415):
  run=f'v15_seed_{seed}';path=source/run/'adapter'
  h=hashlib.sha256((path/'adapter_model.safetensors').read_bytes()).hexdigest()
  assert h==all_results[run+'/replication_complete.json']['adapter_sha256']
  set_seed(seed);tok=AutoTokenizer.from_pretrained(path)
  model=AutoModelForCausalLM.from_pretrained(STUDENT,revision=BASE_REVISION,token=token,device_map={'':0},torch_dtype=torch.float16,quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
  model=PeftModel.from_pretrained(model,path)
  summary[run]=evaluate(model,tok,anchor['mc']['questions'],ROOT/f'{run}_mc.json')
  rows=json.loads((ROOT/f'{run}_mc.json').read_text());assert len(rows)==240
  assert len({(r['id'],r['shift']) for r in rows})==240
  for r in rows:assert r['expected']==expected[(r['id'],r['shift'])] and r['correct']==(parse_answer(r['raw'])==r['expected'])
  all_results[run+'/anchor_mc_original_runner.json']=all_results[run+'/anchor_mc.json']
  all_results[run+'/anchor_mc.json']=rows
  print('HARMONIZED MC',run,json.dumps(summary[run]),flush=True)
  del model;gc.collect();torch.cuda.empty_cache()
 save(ROOT/'summary.json',summary)
 save(ROOT/'all_results.json',all_results)
 save(ROOT/'provenance.json',dict(training_updates=0,teacher_calls=0,reason='Use direct PeftModel inference consistently; original v15 runner prepares kbit model during its own MC evaluation',original_training_output_preserved=True))
 print('HARMONIZED MC COMPLETE',flush=True)
if __name__=='__main__':main()
