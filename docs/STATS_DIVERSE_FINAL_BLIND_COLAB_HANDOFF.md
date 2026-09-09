# Reproduce the Final Comparison in Colab

The completed run scored 45/180 for the original Llama 3.2 3B Instruct model and 170/180 for step180. Reproduction does not constitute a new blind test. Use a GPU runtime and a Hugging Face account with access to the base model.

## Run in one Colab cell

This downloads only the adapter from the pinned Hugging Face revision, reads the pinned evaluation code and questions, and runs both models. Enter your token at the hidden prompt. No training is performed.

```python
import os
import sys
import subprocess
from pathlib import Path
from getpass import getpass

os.environ["HF_TOKEN"] = getpass("Hugging Face token: ").strip()
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

def run(args):
    subprocess.run(args, check=True)

run([sys.executable, "-m", "pip", "install", "-q",
     "transformers==4.57.6", "huggingface_hub==0.36.0",
     "peft>=0.12,<1", "accelerate>=0.34,<2",
     "bitsandbytes>=0.44,<1", "safetensors", "datasets", "sympy"])

repo = Path("/content/3beethoven_reproduction")
code_revision = "881f9aa9091b62b7d4c243c1e78d2bc69eca9b15"
if not repo.exists():
    run(["git", "clone", "https://github.com/Trina0224/3Beethoven.git", str(repo)])
run(["git", "-C", str(repo), "fetch", "origin", code_revision])
run(["git", "-C", str(repo), "checkout", "--detach", code_revision])

# Use a fresh process to avoid mixing notebook-cached package versions.
run([sys.executable, "-c", 'from huggingface_hub import snapshot_download\nsnapshot_download(repo_id="kozakurayuki/3Beethoven-step180",\n revision="682d5555b0e6115135ac7b9b5d718d2abef186de",\n local_dir="/content/3beethoven_step180",\n allow_patterns=["adapter_model.safetensors", "adapter_config.json"])\n'])

command = [sys.executable, "-u",
    str(repo / "scripts/evaluate_stats_diverse_final_blind_colab.py"),
    "--adapter", "/content/3beethoven_step180",
    "--output", "/content/3beethoven_final_blind_eval"]
process = subprocess.Popen(command, stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, text=True, env=os.environ.copy())
for line in process.stdout:
    print(line, end="", flush=True)
if process.wait() != 0:
    raise RuntimeError("Evaluation failed; see the actual error printed above.")

import shutil
from google.colab import files
archive = shutil.make_archive("/content/final_blind_eval_results", "zip",
    root_dir="/content/3beethoven_final_blind_eval")
files.download(archive)
```

## Outputs and known environment issues

The output directory contains raw answers for both models, their summaries, `comparison_summary.json`, and `contract.json`. Expected paired counts are `0->0: 8`, `0->1: 127`, `1->0: 2`, `1->1: 43`. The student has zero pending answers.

The evaluator deliberately uses the pinned base tokenizer. The old adapter archive named a `TokenizersBackend` class incompatible with the tested Transformers installation. Do not restore that tokenizer-loading behavior. Avoid an unrestricted upgrade of `huggingface_hub`: Transformers 4.57.6 requires a version below 1.0. Setting `HF_TOKEN` in Python's `os.environ` makes it available to subprocesses; a separate `!export` notebook cell does not reliably do so.

Only tokenizer compatibility was fixed for the successful run. This documentation update was checked for Python syntax; it was not used to launch another GPU evaluation. Dependency ranges outside the two pinned compatibility packages are not a complete environment lock.

[Final report](STATS_DIVERSE_FINAL_BLIND_RESULTS.md) · [Preserved Traditional Chinese version](STATS_DIVERSE_FINAL_BLIND_COLAB_HANDOFF.zh-TW.md)
