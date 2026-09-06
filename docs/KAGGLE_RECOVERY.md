# Recovering saved statistics experiments

Use individual cells. Do not use Run All on the historical notebook: earlier cells perform paid teacher generation and old training stages.

## Preserved checkpoints

| Version | Script version ID | Saved output |
|---|---|---|
| 5 | 347584475 | v0.3 adapter, teacher corpus, holdout, both ZIP archives |
| 6 | 347586668 | 60-question four-rotation diagnostic and ZIP |
| 7 | 347590242 | v0.4 adapter, reused teacher records, expanded targets, 1,008 responses and ZIP |

The latest output is not a cumulative copy of all older output files. A notebook input added without a pinned version can resolve to the latest version after a restart.

## Restore the original training source

If this notebook's latest output is already attached as an input, remove that input first through its menu. Then run:

```python
import kagglehub
from pathlib import Path
restored = Path(kagglehub.notebook_output_download(
    "trinashih/3beethoven-v0-2/versions/5"
))
assert list(restored.rglob("teacher_train.jsonl"))
assert list(restored.rglob("adapter_model.safetensors"))
```

This uses the [official KaggleHub notebook-output API](https://github.com/Kaggle/kagglehub#download-notebook-outputs). In this session, requesting version 5 while version 6 remained mounted returned the existing mount without the needed files. Removing the current input and repeating the pinned request restored the corpus successfully. Always validate file presence and artifact hashes, rather than treating a returned directory path as successful recovery.

`scripts/run_stats_v0_4.py` pins version 5 and checks the original adapter SHA-256 and exact training/validation question IDs before training. It reads only HF_TOKEN; no new teacher request is made.

## Resume v0.4 in the same working directory

```python
import os, subprocess, sys
subprocess.run(
    [sys.executable, "-u", "scripts/run_stats_v0_4.py"],
    cwd="/kaggle/working/3Beethoven",
    env=dict(os.environ, CUDA_VISIBLE_DEVICES="0", TOKENIZERS_PARALLELISM="false"),
    check=True,
)
```

The runner skips existing responses and resumes a saved training checkpoint when available. It refuses protocol mismatches. Resuming after a destroyed session additionally requires restoring that run's working directory from its saved output; merely rerunning does not recover unsaved files.

After completion, run `scripts/verify_stats_v0_4.py`, then Quick Save with output saving enabled. Confirm the saved version is Successful and its ZIP is present before stopping the GPU session. Model ZIPs remain on Kaggle; code, reports and raw text results are committed to GitHub.
