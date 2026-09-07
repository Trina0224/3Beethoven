"""Six preregistered retrainings; immutable historical runners, separate outputs."""
import os,sys,json,gc,hashlib,shutil,subprocess,types,zipfile,importlib.metadata
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/kaggle/working/3beethoven_seed_replication')
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def worker(family,seed):
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,set_seed
 from peft import PeftModel
 from kaggle_secrets import UserSecretsClient
 from run_stats_v0_4 import BASE_REVISION,evaluate as mc
 from run_stats_v0_19 import evaluate as formula
 from flight_run_stats_v0_3 import STUDENT
 folder=ROOT/f'{family}_seed_{seed}';folder.mkdir(exist_ok=True)
 source=REPO/'scripts'/f'run_stats_v0_{15 if family=="v15" else 19}.py'
 code=source.read_text().replace('1515' if family=='v15' else '1919',str(seed))
 if family=='v15':
  old="(('baseline',None),('v14',V14/'adapter'),('v15',ROOT/'adapter'))"
  assert old in code;code=code.replace(old,"(('v15',ROOT/'adapter'),)")
 module=types.ModuleType(f'replica_{family}_{seed}');module.__file__=str(source)
 exec(compile(code,str(source),'exec'),module.__dict__)
 module.ROOT=folder
 if family=='v15':
  rows={s:read(ROOT/'frozen'/f'v15_{s}_examples.json') for s in ('train','validation')}
  module.teacher_rows=lambda repo:rows
 module.main()
 if family=='v15':adapter=folder/'adapter'
 else:
  sel=read(folder/'selection.json');epoch=sel['selected_epoch'] or sel['diagnostic_epoch'];adapter=folder/f'epoch_{epoch}'/'adapter'
 set_seed(seed)
 token=UserSecretsClient().get_secret('HF_TOKEN')
 tok=AutoTokenizer.from_pretrained(adapter)
 model=AutoModelForCausalLM.from_pretrained(STUDENT,revision=BASE_REVISION,token=token,device_map={'':0},torch_dtype=torch.float16,quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
 model=PeftModel.from_pretrained(model,adapter)
 anchor=read(REPO/'docs/STATS_PERMANENT_ANCHOR_V1.json');qs=read(REPO/'docs/STATS_V19_PERTURBATION_QUESTIONS.json')
 # The original v19 runner already evaluated exactly the permanent formulation questions.
 if family=='v19':
  for split in ('old_test','new_test'):shutil.copy2(folder/f'epoch_{epoch}_{split}.json',folder/f'anchor_{split}.json')
 else:
  data=read(REPO/'docs/STATS_V0_19_FROZEN_QUESTIONS.json')
  for split in ('old_test','new_test'):formula(model,tok,data[split],folder/f'anchor_{split}.json')
 formula(model,tok,qs,folder/'perturbation.json')
 if family=='v15':shutil.copy2(folder/'v15_old.json',folder/'anchor_mc.json')
 else:mc(model,tok,anchor['mc']['questions'],folder/'anchor_mc.json')
 del model;gc.collect();torch.cuda.empty_cache()
 import safetensors.torch
 weights=safetensors.torch.load_file(str(adapter/'adapter_model.safetensors'),device='cpu')
 assert all(torch.isfinite(t).all().item() for t in weights.values())
 result={'family':family,'seed':seed,'adapter_sha256':sha(adapter/'adapter_model.safetensors'),'finite_tensors':len(weights),'training':read(folder/'training_complete.json'),'automatic':{}}
 for name,n in [('anchor_old_test',96),('anchor_new_test',96),('perturbation',88),('anchor_mc',240)]:
  rows=read(folder/f'{name}.json');assert len(rows)==n
  result['automatic'][name]={'n':n,'correct':sum(r['correct'] for r in rows),'pending':sum(r.get('review_required',False) for r in rows)}
 save(folder/'replication_complete.json',result)
 print('SEED COMPLETE',family,seed,json.dumps(result),flush=True)
def main():
 ROOT.mkdir(exist_ok=True)
 if len(sys.argv)>1:return worker(sys.argv[1],int(sys.argv[2]))
 import torch
 assert torch.__version__.split('+')[0]=='2.10.0'
 import kagglehub
 saved=Path('/kaggle/input/notebooks/trinashih/3beethoven-v0-2')
 for v,h in [(15,'9369d52de4a886df9da0c872cd41bd4e01af0a38bf02ad724b5951c1a6b9f5d3'),(14,'c7def77757fefaaf41db6938500159795a47503dac54d72d79113de47a3239a5')]:
  name=f'3beethoven_stats_v0_{v}';candidates=[p for p in saved.rglob(name) if (p/'adapter/adapter_model.safetensors').exists()]
  assert candidates,f'Missing v{v}'
  src=min(candidates,key=lambda p:len(p.parts));assert sha(src/'adapter/adapter_model.safetensors')==h
  dst=Path('/kaggle/working')/name;shutil.copytree(src/'adapter',dst/'adapter',dirs_exist_ok=True)
  if v==15:
   for split,n in [('train',200),('validation',30)]:
    rows=read(src/f'{split}_examples.json');assert len(rows)==n
    save(ROOT/'frozen'/f'v15_{split}_examples.json',rows)
 manifest={'seeds':[1919,2027,31415],'families':['v15','v19'],'teacher_calls':0,'selection':'Original validation rules only; all runs retained; no test-based seed selection','packages':{p:importlib.metadata.version(p) for p in ('torch','transformers','peft','bitsandbytes','datasets','accelerate')},'gpu':torch.cuda.get_device_name(0),'sources':{str(p.relative_to(REPO)):sha(p) for p in list((REPO/'scripts').glob('*.py'))+list((REPO/'docs').glob('*ANCHOR*.json'))},'frozen_v15_data':{p.name:sha(p) for p in (ROOT/'frozen').glob('*.json')},'parents':{str(v):sha(Path(f'/kaggle/working/3beethoven_stats_v0_{v}/adapter/adapter_model.safetensors')) for v in (14,15)}}
 save(ROOT/'execution_manifest.json',manifest)
 shutil.copytree(REPO/'scripts',ROOT/'source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 shutil.copy2(REPO/'docs/STATS_SEED_REPLICATION_PROTOCOL.md',ROOT/'protocol.md')
 print('SEED MANIFEST FROZEN',json.dumps(manifest),flush=True)
 for family in manifest['families']:
  for seed in manifest['seeds']:
   subprocess.run([sys.executable,__file__,family,str(seed)],cwd=REPO,check=True)
 results={str(p.relative_to(ROOT)):read(p) for p in ROOT.rglob('*.json') if 'source' not in p.parts and 'checkpoints' not in p.parts}
 save(ROOT/'all_results.json',results)
 archive=ROOT.with_suffix('.zip')
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for p in ROOT.rglob('*'):
   if p.is_file() and 'checkpoints' not in p.parts:z.write(p,str(p.relative_to(ROOT)))
 with zipfile.ZipFile(archive) as z:assert z.testzip() is None
 save(ROOT.parent/'seed_replication_backup_receipt.json',dict(bytes=archive.stat().st_size,sha256=sha(archive),crc_verified=True,optimizer_checkpoints='preserved in formal output; excluded from compact zip'))
 print('ALL SIX TRAINING RUNS COMPLETE',flush=True)
if __name__=='__main__':main()
