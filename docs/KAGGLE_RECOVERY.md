# Recovering saved statistics experiments

Use individual cells. Do not use Run All on the historical notebook: earlier cells perform paid teacher generation and old training stages.

## Preserved checkpoints

| Version | Script version ID | Saved output |
|---|---|---|
| 5 | 347584475 | v0.3 adapter, teacher corpus, holdout, both ZIP archives |
| 6 | 347586668 | 60-question four-rotation diagnostic and ZIP |
| 7 | 347590242 | v0.4 adapter, reused teacher records, expanded targets, 1,008 responses and ZIP |
| 8 | 347596444 | v0.5 audited 204-record corpus, original teacher calls and revisions |
| 9 | 347598932 | v0.5 selected adapter, corpus, 1,152 responses, audit documents and verified ZIP |
| 10 | 347602173 | v0.6 paired corpus, rejected abstract cards and teacher test |
| 11 | 347605195 | v0.6 selected adapter, 1,296 responses, logs and verified ZIP |

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

## Restore completed v0.5 without teacher calls

Version 9 is confirmed Successful; the GPU session was stopped after preservation. Use the saved ZIP to restore the selected step-45 model and its completed results. Do not rerun historical generation cells or Run All.

Remove an existing input for this same notebook before requesting version 9, as explained above. In a fresh working directory:

```python
from pathlib import Path
import hashlib, shutil, zipfile, kagglehub

saved = Path(kagglehub.notebook_output_download(
    "trinashih/3beethoven-v0-2/versions/9"
))
archive = Path("/kaggle/working/3beethoven_stats_v0_5.zip")
source = saved / archive.name
expected = "370591a3110b0b90efbfea06ab7db67009d955072879fde6fe28bd6742f0e2f2"
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
shutil.copy2(source, archive)
root = archive.with_suffix("")
assert not root.exists(), "Use a fresh directory; do not mix an existing run"
root.mkdir()
with zipfile.ZipFile(archive) as z:
    z.extractall(root)  # Exact verified experiment archive above.
```

With the GitHub repository available at `/kaggle/working/3Beethoven`, run `scripts/verify_stats_v0_5.py` to check stored results, tensor finiteness and the original ZIP manifest. This verifier makes no teacher requests and does not train. The selected adapter is under `3beethoven_stats_v0_5/adapter/`; the base revision and hashes are in the [result report](STATS_V0_5_RESULTS.md).

If rerunning the comparison runner is necessary, first preserve the restored working files, remove the version-9 input, and mount version 5 for the original v0.3 comparison adapter. `run_stats_v0_5.py` requires that original adapter hash and skips saved responses and completed training. Merely mounting the latest notebook output is insufficient. Do not rerun the finalizer just to inspect the original ZIP: updating provenance and source snapshots would intentionally create a different archive hash.

Raw model responses, full logs, paired comparisons and reproduction differences are also available in [STATS_V0_5_RESULTS.json](STATS_V0_5_RESULTS.json). The model archive stays on Kaggle; it does not need to be regenerated to retrieve it.

## Restore completed v0.6 without teacher calls

Version 11 is confirmed Successful, its ZIP was present in saved output, and the GPU session was stopped. v0.6 did not improve fresh accuracy; v0.5 remains the leading experimental candidate. Recovery does not require another training run.

Remove any existing input for this same notebook before requesting pinned version 11. In a fresh working directory:

```python
from pathlib import Path
import hashlib, shutil, zipfile, kagglehub

saved = Path(kagglehub.notebook_output_download(
    "trinashih/3beethoven-v0-2/versions/11"
))
archive = Path("/kaggle/working/3beethoven_stats_v0_6.zip")
source = saved / archive.name
expected = "903d4cc86ea6ce97a47b5a7afbbb5abdef1667c256266d90ed816f1bb2844d5d"
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
shutil.copy2(source, archive)
root = archive.with_suffix("")
assert not root.exists(), "Use a fresh directory; do not mix an existing run"
root.mkdir()
with zipfile.ZipFile(archive) as z:
    z.extractall(root)  # Exact verified archive above.
```

With the repository at `/kaggle/working/3Beethoven` and its verification dependencies installed, run `scripts/verify_stats_v0_6.py`. It verifies stored responses, finite tensors and archive checksums on CPU; it does not train, load the base model or query the teacher. The selected step-45 adapter is in `3beethoven_stats_v0_6/adapter/`. See the [complete result report](STATS_V0_6_RESULTS.md) for hashes and limitations.

Do not run `generate_stats_v0_6.py`: its paid abstract-card entry point is retired because those cards failed audit. Do not rerun preparation simply to inspect saved results; absent caches, preparation can call the teacher. The saved archive already includes the completed paired corpus and API ledger.

Only if the full comparison runner is needed: preserve the restored v0.6 working directory, remove the version-11 notebook input, and mount pinned version 9 for the required v0.5 comparison adapter. `run_stats_v0_6.py` checks its hash and skips completed results/training. The latest notebook output alone does not supply that control adapter. Preserve original archives rather than repacking them just to inspect results.

Code and complete text results are in GitHub; the model ZIP remains in Kaggle version 11. Its source snapshot records the code used for the experiment; later GitHub changes add reporting and safer recovery without changing the selected adapter.
