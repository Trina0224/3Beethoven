# Run only after mounting the verified seed preparation output.
from pathlib import Path
import os,sys,shutil,subprocess
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
source=Path('/kaggle/input/notebooks/trinashih/3beethoven-v0-2/seed_repo')
assert source.is_dir(),'Mount the completed seed preparation output'
repo=Path('/kaggle/working/seed_repo')
shutil.copytree(source,repo,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','.git'))
subprocess.run([sys.executable,'-m','pip','install','--quiet','transformers==5.0.0','peft==0.19.1','bitsandbytes==0.50.2','datasets==5.0.0','accelerate==1.13.0'],check=True)
subprocess.run([sys.executable,str(repo/'scripts/run_training_seed_replication.py')],cwd=repo,check=True)
