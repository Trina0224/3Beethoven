"""Inference-only analysis of frozen v15 and v19; no optimization or teacher API."""
import os,json,gc,hashlib,shutil,importlib.metadata,zipfile
from pathlib import Path
from flight_run_stats_v0_3 import read_json as read,save_json as save,package,STUDENT
from run_stats_v0_4 import BASE_REVISION,evaluate as evaluate_mc
from run_stats_v0_19 import evaluate as evaluate_formula
from stats_curriculum_v0_19 import score,prompt,digest
from stats_v0_3_common import parse_answer
ROOT=Path('/kaggle/working/3beethoven_v19_analysis')
INPUT=Path('/kaggle/input/notebooks/trinashih/3beethoven-v0-2')
HASHES={'v15':'9369d52de4a886df9da0c872cd41bd4e01af0a38bf02ad724b5951c1a6b9f5d3','v19':'9fe491ee18bd6e0b9da03ab0fa6b991d583cf6041364f4a742e00fd9f7f75a18'}
def main():
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,set_seed
 from peft import PeftModel
 from kaggle_secrets import UserSecretsClient
 repo=Path(__file__).resolve().parents[1];ROOT.mkdir(exist_ok=True)
 qs=read(repo/'docs/STATS_V19_PERTURBATION_QUESTIONS.json');anchor=read(repo/'docs/STATS_PERMANENT_ANCHOR_V1.json')
 assert digest(qs)=='2e562ae07c0cd22236136e522dd83420d43e20a1519c9bde32a4b0a59f814536'
 assert digest(anchor)=='ee141365e07e5b838c40e270d22276bc02e17cf85f884f7b6074141d1951b3c0'
 for name,h in anchor['source_hashes'].items():assert hashlib.sha256((repo/'scripts'/name).read_bytes()).hexdigest()==h,name
 assert torch.__version__.split('+')[0]=='2.10.0',torch.__version__
 save(ROOT/'questions.json',qs);save(ROOT/'anchor.json',anchor)
 shutil.copy2(repo/'docs/STATS_V19_ANALYSIS_PROTOCOL.md',ROOT/'protocol.md')
 shutil.copytree(repo/'scripts',ROOT/'source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 save(ROOT/'environment.json',dict(packages={k:importlib.metadata.version(k) for k in ('torch','transformers','peft','bitsandbytes','datasets','accelerate')},base_revision=BASE_REVISION,adapter_hashes=HASHES,teacher_calls=0,training_updates=0))
 token=UserSecretsClient().get_secret('HF_TOKEN');paths={'v15':INPUT/'3beethoven_stats_v0_15/adapter','v19':INPUT/'3beethoven_stats_v0_19/epoch_1/adapter'}
 tok=AutoTokenizer.from_pretrained(paths['v15']);summary={}
 for name,path in paths.items():
  assert hashlib.sha256((path/'adapter_model.safetensors').read_bytes()).hexdigest()==HASHES[name]
  set_seed(1919)
  model=AutoModelForCausalLM.from_pretrained(STUDENT,revision=BASE_REVISION,token=token,device_map={'':0},torch_dtype=torch.float16,quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
  model=PeftModel.from_pretrained(model,path);summary[name]={}
  summary[name]['formula']=evaluate_formula(model,tok,qs,ROOT/f'{name}_formula.json')
  rows=read(ROOT/f'{name}_formula.json');qmap={q['id']:q for q in qs}
  def count(selected):return dict(n=len(selected),correct=sum(r['correct'] for r in selected),pending=sum(r.get('review_required',False) for r in selected))
  summary[name]['by_variant']={v:count([r for r in rows if qmap[r['id']]['variant']==v]) for v in ('original','paraphrase','magnitude','exactly_one','at_least_one')}
  print('ANALYSIS FORMULA',name,json.dumps(summary[name]['by_variant']),flush=True)
  summary[name]['mc']=evaluate_mc(model,tok,anchor['mc']['questions'],ROOT/f'{name}_mc.json')
  print('ANALYSIS MC',name,summary[name]['mc']['overall'],summary[name]['mc']['all_four_correct'],flush=True)
  del model;gc.collect();torch.cuda.empty_cache();save(ROOT/'summary.json',summary)
 pending=[];count=0
 for name in paths:
  rows=read(ROOT/f'{name}_formula.json');assert len(rows)==88
  for r in rows:
   q=qmap[r['id']];assert r['question_sha256']==digest(q) and r['prompt']==prompt(q)
   assert r['correct']==score(r['raw'],q)['correct']
   if r.get('review_required'):pending.append(dict(model=name,**r,reference=q['expression'],question=q['question']))
  mc=read(ROOT/f'{name}_mc.json');assert len(mc)==240
  expected={(r['id'],r['shift']):r['expected'] for r in anchor['mc']['rotations']}
  assert len({(r['id'],r['shift']) for r in mc})==240
  for r in mc:assert r['expected']==expected[(r['id'],r['shift'])] and r['correct']==(parse_answer(r['raw'])==r['expected'])
  count+=len(rows)+len(mc)
 assert count==656;save(ROOT/'verification.json',dict(responses=count,pending=pending,adapter_hashes=HASHES))
 package(ROOT);archive=ROOT.with_suffix('.zip')
 with zipfile.ZipFile(archive) as z:assert z.testzip() is None
 save(ROOT.parent/'v19_analysis_backup_receipt.json',dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),crc_verified=True))
 print('ANALYSIS COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':
 os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['TOKENIZERS_PARALLELISM']='false';main()
