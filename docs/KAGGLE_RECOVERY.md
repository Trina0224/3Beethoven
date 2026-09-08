# 最後成果模型恢復

2026-09-07 22:15 PDT（America/Los_Angeles）

**最後採用的成功範例：`V57-repeat_control`。** 這是展示名稱；實際權重資料夾仍是 `repeat_control/adapter`，沒有重新命名權重，也沒有新增一個學生。

它是從原 v15 接續訓練、使用修正後 Llama 老師教材的 Llama-3.2-3B-Instruct 學生。在直接問法的雙事件機率列式套題中，原 v15 **12/16**，本學生 **16/16**。學生自行列出代入數值的算式，計算交給程式。

這是使用者收束目標後選定的既有成果展示：題組已曝光，屬回顧性範例，不是新盲測或通用統計能力認證。所有 16 題均列入；四組機率各問四種事件。機率組合未出現在該學生的事件教材中。原廣泛保留門檻未通過的紀錄維持不變。

## 正確下載與載入對象

- [下載已保存的模型 ZIP](https://www.kaggle.com/code/trinashih/3beethoven-v0-2/output?scriptVersionId=348126445&select=3beethoven_semantic_course_students.zip)（Kaggle 權限與 Hugging Face 基礎模型存取權需沿用既有帳號）。
- ZIP：`3beethoven_semantic_course_students.zip`；選用其中 **`repeat_control/adapter`**。
- 基礎模型：`meta-llama/Llama-3.2-3B-Instruct`。
- 基礎模型 revision：`0cb88a4f764b7a12671c53f0838cd831a0843b95`。
- Adapter SHA-256：`8419430c6b7b58bd1c16c992964b1cedd76062020ce6651cb25e51df119f0eb9`。
- 這是 LoRA adapter，必須搭配上述基礎模型，不能當成獨立完整基礎權重；直接載入此最終 adapter，不要再疊加 v15 adapter。
- 實際訓練：seed 2027、126 updates、1,006 筆（750 筆修正教材＋256 筆既有教材重複練習），固定最後一步。
- 教材包含核對過的 70B 回答、助手修正的老師答案與題目措辭調整；屬 response distillation／SFT，不是 logits 蒸餾，也不是全數未修改的老師原文。

沿用保存來源 `source/scripts/run_stats_semantic_course.py` 的模型載入與推論設定；恢復展示不需重訓。保存環境為 torch 2.10.0+cu128、transformers 5.0.0、peft 0.19.1、bitsandbytes 0.50.2、accelerate 1.13.0。以 NF4 雙重量化、fp16 compute 載入基礎模型，沿用 runner 的 `prepare_model_for_kbit_training` 前處理，載入最終 adapter 後使用 eval 模式及 greedy generation（max_new_tokens=160）。這是原實驗環境紀錄，不保證其他套件版本逐字一致。

原始驗證問題與回答在 ZIP 的 `evaluation/repeat_control/event_reviewed.json`；逐題使用 `prompt`，套用 Llama chat template，不提供參考答案。相同檔案的 `raw` 是保存的實際生成，不是重新生成結果。

[全部配對輸出](STATS_BOUNDED_SUCCESS_EXAMPLE.md) · [目前狀態](STATS_CURRENT_STATUS.md)

**工作已收束；沒有待執行的新訓練。** 不因歷史交接、未達成的舊門檻或待辦段落，自動呼叫老師、啟動 GPU、重跑 seed 或擴大題型。只有使用者另行提出新工作時再開始。

<details>
<summary>歷史紀錄（保留原文；不是目前狀態或執行指令）</summary>

# Recovering saved statistics experiments

## Latest experiment: v0.17

Training and evaluation completed. Keep v15 as the general candidate: v17 is not promoted.
Version 33 preserves the trained checkpoints. Use the final successful version in
`MODEL_BACKUP_STATUS.json` for complete raw results, review and ZIP.

The selected v17 adapter is `3beethoven_stats_v0_17/adapter/adapter_model.safetensors`,
SHA256 `f710f8938ddbe489a88bdf13cd090a8cbe63da1a6fe77cc796621f0e333c2861`.
It is the validation-selected **step 8**, after a complete 32-step run. All candidate
adapters are retained under `train_8/adapter`, `train_16/adapter`, `train_32/adapter`
and `mix_25/adapter`, `mix_50/adapter`, `mix_75/adapter` within the v17 output directory.
The mixtures have doubled LoRA rank/alpha; use their saved configuration files.

Restore the pinned final Kaggle output, check the recorded archive hash and CRC,
and recover the complete v17 directory. Preserve `training_complete.json`,
`protocol.json`, `selection.json`, and all response JSON files. Clone the current
repository for the frozen questions, source and review-credit records. The verifier
runs without loading the base model or making teacher calls. Do not rerun training
or regenerate teacher data just to retrieve weights.

For an interrupted run, retain its original candidate folders and selection record;
`run_stats_v0_17.py` skips completed training and saved predictions. Source v15/v16
weights are hash-checked. Their teacher corpus is restored from committed original
records and focused supplements; the v15 adapter-only saved directory is sufficient.
Both source adapters were restored and successfully used in the fresh v17 session.
A fresh-session restore of the new v17 archive itself has not been tested.

## Previous candidate: v0.16

Version 31 preserves completed training. **Version 32 is Successful** and contains
the complete predictions and verification; restore `trinashih/3beethoven-v0-2/versions/32`
with `kagglehub.notebook_output_download`. Locate
`3beethoven_stats_v0_16/adapter/adapter_model.safetensors` and verify SHA-256
`117a009f72ebafe6e6baefef62a6b81e7bbcefbc902f7eb3d93f5e73f48d46d0`.
Final ZIP `3beethoven_stats_v0_16.zip`: 371,440,108 bytes, SHA-256
`494fd49d497ab1569578eb4983d22d90bd2721495bb1d683f46b3a26c385067f`.
It includes intermediate and final adapters. Use the top-level `adapter` for the
selected student. v16 is a rule-assisted candidate, not a general replacement for v15.
The base revision is pinned in `protocol.json`; existing HF authorization is needed.

## Retained general candidate: v0.15, saved Version 29

Restore `trinashih/3beethoven-v0-2/versions/29` with
`kagglehub.notebook_output_download`, then locate
`3beethoven_stats_v0_15/adapter/adapter_model.safetensors`.
Verify SHA-256 `9369d52de4a886df9da0c872cd41bd4e01af0a38bf02ad724b5951c1a6b9f5d3`.
ZIP `3beethoven_stats_v0_15.zip`: 93,892,056 bytes,
SHA-256 `49b1e411edab050f0aae694b0b1b80181613bca5afacc8d79dd75abfe0eed097`.
Version 29 was confirmed Successful with output saving enabled. The adapter needs
the base model revision in its training protocol; it is not a standalone base model.


## Previous student: v0.14, saved Version 28

Version 28 was confirmed Successful with output saving enabled. Restore specifically
`trinashih/3beethoven-v0-2/versions/28` using the existing `kagglehub.notebook_output_download`
workflow, then locate `3beethoven_stats_v0_14/adapter/adapter_model.safetensors`.
Verify SHA-256 `c7def77757fefaaf41db6938500159795a47503dac54d72d79113de47a3239a5` before loading.
The ZIP is `3beethoven_stats_v0_14.zip` (92,760,672 bytes),
SHA-256 `e360452b023423a956c1a43a1284fe9d7b749acb847284cab26ffd1072fe7e53`.
The adapter requires the original Llama 3.2 3B Instruct base revision recorded in
`training_protocol.json`; it is not a standalone full base model.
Do not assume the latest notebook input contains older versions' weights.


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
| 12 | 347608100 | Inference-only diagnostic, 576 responses and verified ZIP; no model weights |
| 13 | 347615356 | v0.8 diagnostic and v0.9 audited preparation, caches and ZIPs; no v0.9 trained weights |
| 14 | 347620387 | v0.9 selected adapter, 960 student responses, teacher caches, verified ZIP and v0.8 diagnostic |

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

## Restore v0.7 diagnostic results

Kaggle version 12 (347608100) is confirmed Successful and its diagnostic ZIP was present before stopping the GPU. The archive is `3beethoven_stats_diagnostic_v0_7.zip`, 32,976 bytes, SHA-256 `3bf4a8d136fbf27d2df093708a83c31bda5c9161a24bafdf8c0b9d549216527d`.

Remove an existing same-notebook input before requesting `trinashih/3beethoven-v0-2/versions/12`. Verify the ZIP hash, copy the ZIP to /kaggle/working, and extract into a fresh /kaggle/working/3beethoven_stats_diagnostic_v0_7 directory. Run `scripts/verify_stats_diagnostic_v0_7.py` from the repository. Verification needs no GPU, teacher requests or training.

To reproduce inference, use `scripts/diagnose_stats_v0_7.py` with pinned version 9's v0.5 adapter after removing a conflicting same-notebook input. Version 12 has diagnostic data, not replacement weights. The script checkpoints answers and refuses protocol mismatches. Avoid Run All.

The [diagnostic report](STATS_DIAGNOSTIC_V0_7_RESULTS.md) and [full JSON with explicit format review](STATS_DIAGNOSTIC_V0_7_RESULTS.json) are preserved in GitHub. Strict original scores remain unchanged; the report distinguishes formatting corrections from truncated calculations.

## v0.8 diagnostic and v0.9 preparation

Version 13 output was verified to contain `3beethoven_stats_diagnostic_v0_8.zip` and `3beethoven_stats_v0_9.zip`. v0.8 ZIP SHA-256: `e0662dd2179965909f1463fa74888b4443289ff647cb705ed813d4e6bc296107` (19,500 bytes). v0.9 preparation is not a completed model. Its approved corpus is also preserved in GitHub with original rule and numerical-solution provenance.

Do not rerun the v0.9 generator after applying audited rules: it can restore the original pre-audit rule lines. Restore the preparation archive and audit files instead. The runner validates their exact hashes and teacher-cache provenance before training. To resume it, preserve its working directory and mount pinned version 9 separately for the v0.5 control adapter, using the same input-conflict precautions above.

## Restore completed v0.9

Version 14 (347620387) is confirmed Successful; its saved output contains `3beethoven_stats_v0_9.zip` and the v0.8 diagnostic ZIP. [Open version 14 output](https://www.kaggle.com/code/trinashih/3beethoven-v0-2/output?scriptVersionId=347620387). The current result is a targeted-skill improvement with a remaining robustness failure; keep the version-9 v0.5 control.

Remove any existing same-notebook input before requesting pinned version 14. Recovery needs no new teacher calls:

```python
from pathlib import Path
import hashlib, shutil, zipfile, kagglehub
saved = Path(kagglehub.notebook_output_download(
    "trinashih/3beethoven-v0-2/versions/14"
))
archive = Path("/kaggle/working/3beethoven_stats_v0_9.zip")
source = saved / archive.name
assert hashlib.sha256(source.read_bytes()).hexdigest() == "ddc3f29f1533acf037ad01077875650e7309ed81148cd3ec8cd20ef40865fdcc"
shutil.copy2(source, archive)
root = archive.with_suffix("")
assert not root.exists(), "Use a fresh directory"
root.mkdir()
with zipfile.ZipFile(archive) as z:
    z.extractall(root)  # Exact verified archive above.
```

ZIP bytes: 93,446,957. Adapter SHA256: `805a2170a805f6176aa3837857890b8c44fc8f854d16cbc3085ae220e5502c7c`. Selected checkpoint: step135. With repository scripts and verification dependencies installed, run `scripts/verify_stats_v0_9.py`: it verifies saved responses, finite tensors and the ZIP manifest on CPU, without training, base-model loading or teacher requests. Keep the original ZIP unchanged.

The ZIP preserves raw strict results, teacher provenance, source and training state. The later independent format audit and narrative report are in [GitHub results JSON](STATS_V0_9_RESULTS.json) and [report](STATS_V0_9_RESULTS.md), also copied into the repository folder in saved output before final preservation; later GitHub reporting additions do not alter the ZIP. Do not regenerate data or train merely to recover the completed adapter.

After version-14 output verification, the GPU session was stopped and the editor explicitly showed `Draft Session off (run a cell to start)`.


## v0.10 audited preparation checkpoint

Kaggle version 15 (347625570) is Successful and its output visibly contains `3beethoven_stats_v0_10.zip` (551,081 bytes; SHA-256 `f5772cb391f1bbd342a0e3f278e39a1a2bbe554ce72b215af139d9dbb0d2fb6c`). This is preparation only: no v0.10 student training or test has run. The 112 audited teacher records yield 516 training and 64 validation sequences with rehearsal; actual maximum length is 303 tokens, below the 768-token cap. Corpus and audit are committed separately. Teacher usage: 221 calls, $0.02018775, complete cost reporting.

Missing `bitsandbytes==0.50.2` blocks the frozen 4-bit training configuration. Automatic approval review rejected installation and requires action-time installation confirmation. Existing Secrets worked for teacher generation; no new notebook or Secret selection is needed. Complete preparation was saved before requesting that confirmation.

Restore pinned version 15 using `kagglehub.notebook_output_download("trinashih/3beethoven-v0-2/versions/15")` after removing any conflicting same-notebook input. Copy and hash-check the v0.10 ZIP above, then extract into `/kaggle/working/3beethoven_stats_v0_10`; restore the v0.9 adapter from the unchanged version-14 archive using the preceding instructions. Preserve the teacher cache, audit gate and prepared examples. Do not rerun the teacher generator or use Run All. After specific installation approval, install `bitsandbytes==0.50.2`, then run `python -u scripts/run_stats_v0_10.py` from the pulled repository. Run the verifier and independently audit numeric formats, preserve results and stop GPU afterward.


### Recovery verified after restart (2026-09-05 PT)

User explicitly approved bitsandbytes==0.50.2 installation; it succeeded. The fresh session had no working files. v0.10 preparation was restored from the version-15 ZIP and v0.9 from version 15's `3beethoven_stats_v0_9` directory. Version 15 does **not** contain the original v0.9 ZIP; use the directory or version 14's ZIP. The restored adapter is 97,307,544 bytes and its SHA-256 exactly matches the recorded v0.9 hash. See `scripts/restore_stats_v0_10_preparation.py` for a non-overwriting recovery with manifest checks. The v0.10 runner then started same-session baseline evaluation.

GitHub contains the corpus, code, results and recovery instructions. Weight binaries are currently preserved in Kaggle saved outputs, **not committed to GitHub**. A local `git push --dry-run` failed because command-line GitHub authentication is unavailable. Browser download attempts also timed out. Do not describe the GitHub text-file backup as a weight backup. Keep saved Kaggle versions 14 and 15 until an independent binary backup is verified.


An independent v0.9 binary backup subsequently succeeded: browser download events timed out, but both downloaded files arrived and matched the exact adapter SHA-256. `3beethoven_v0_9_weights_backup.zip` (90,115,646 bytes; SHA-256 `773cee2a8262ba7533af87aaeb18ae022b83a92b62dad144dd2ce72ebca08ec1`) was saved separately and contains the original weight bytes plus a clearly labeled reconstructed minimal loading config. It excludes the base model and optimizer state. Original tokenizer and metadata remain in saved Kaggle outputs. This independent backup is not a GitHub binary commit.


## v0.10 completed — 2026-09-05 PDT

Installation was authorized and completed; training and all 1,200 student responses finished. Same-test numeric scores improved 13/48 to 20/48 and MC 67/192 to 87/192; old MC 127/240 to 130/240. Primary improvement/half-correct goals were not met. See [reviewed report](STATS_V0_10_REPORT.md) and complete raw STATS_V0_10_RESULTS.json. Fraction arithmetic remains the principal bottleneck.

Kaggle Quick Save version 18 (script version 347636958) is Successful and contains final 93,104,858-byte 3beethoven_stats_v0_10.zip, SHA-256 470e4013b2f11ef52e6bd60736f73a1121e66e0bfe757093a8d3fd4e9affc677. Selected adapter SHA-256 14812770a7e612ab984e4ffad54bf514a3e00425655aa5adf732b975502f96f9. Restore version 18, not preparation-only version 15. GitHub stores code/data/results; binary backup status is in MODEL_BACKUP_STATUS.json.


## v0.11 selected model saved; evaluation running

Kaggle version20 contains checkpoint-115, including 97.31 MB adapter, optimizer, RNG, scheduler, scaler, trainer state and tokenizer (verified in saved-output UI). Version21 is Successful and contains the selected model after the two-epoch run. Validation selected checkpoint115 (loss0.1145005077), not checkpoint230 (loss approximately0.1325). No test results selected this checkpoint.

If the session is lost before final evaluation, recover version21 with `kagglehub.notebook_output_download('trinashih/3beethoven-v0-2/versions/21')`. Copy its `3beethoven_stats_v0_10` and `3beethoven_stats_v0_11` directories into `/kaggle/working/` in a fresh session before running `scripts/run_stats_v0_11.py`. Preserve `training_complete.json`, `training_protocol.json`, selected `adapter/`, frozen questions, and partial response files; the runner verifies protocol/adapter provenance, skips completed training, and continues missing responses. Do not overwrite a newer working run with an older snapshot. Use the eventual final version in MODEL_BACKUP_STATUS.json once available. GitHub has the fixed data, protocol and scripts; binary weights remain in the saved outputs until the final independent ZIP backup.


## v0.11 completed — 2026-09-06 00:38 PDT

Final Kaggle version 24 (script version ID 347652560) is Successful and supersedes version 21 for recovery. All 816 evaluation responses, independent format review and final report are preserved. Restore pinned version 24 using the same input-conflict precautions above; its v0.10 and v0.11 directories preserve the comparison context. No new teacher calls or training are needed to retrieve results.

The final `3beethoven_stats_v0_11.zip` is 92,805,792 bytes, SHA-256 `3b34b0590d0f3e40a203c2b6dda299e12640e71e53c87dbd56ef5ca465fac348`. It contains the selected checkpoint-115 adapter, exact original metadata, data, results, source, final report and independent review (81 manifest files). Optimizer/checkpoint state remains in full Kaggle saved output. Selected adapter SHA-256: `9994b0eb73cf824791ffbeb81dd08a301bb08801e2c38f38829af3cfd8618541`.

The final ZIP was independently downloaded and saved; archive hash, ZIP CRC and exact adapter hash were verified. The GPU editor explicitly showed `Draft Session off (run a cell to start)` after final preservation. GitHub contains code and full text results, not binary model weights. See [final report](STATS_V0_11_REPORT.md) and [backup status](MODEL_BACKUP_STATUS.json). Results are mixed; retain both v0.10 and v0.11 rather than treating v0.11 as an overall replacement.


## Restore v0.12 selected weights and resume evaluation

Kaggle Quick Save version 25 was confirmed Successful after v0.12 training and archive creation, while the frozen three-model evaluation was still running. Selected checkpoint135 had validation loss0.15432609617710114; the provisional verified-in-run ZIP size was92,802,432 bytes. No final ZIP SHA-256 or independent download is recorded yet.

In a fresh Kaggle working directory, restore `kagglehub.notebook_output_download('trinashih/3beethoven-v0-2/versions/25')`. Copy its `3beethoven_stats_v0_10`, `3beethoven_stats_v0_11` and `3beethoven_stats_v0_12` directories into `/kaggle/working/` without overwriting newer files. Confirm `training_complete.json`, `training_protocol.json`, the selected `adapter/`, frozen questions and partial response checkpoint files are present. Then pull the repository and run `scripts/run_stats_v0_12.py`; the runner validates provenance, skips the completed training marker and saved responses, and continues missing responses.

Version 25 is the selected-weight recovery checkpoint, not the final evaluated archive. After completion, run the verifier, preserve a new final Kaggle version and independently verify the downloaded ZIP before updating hashes or stopping the GPU. The exact observed state and partial baseline are in [STATS_V0_12_INTERIM.md](STATS_V0_12_INTERIM.md).


## v19 formal recovery

The initial detached interactive run was interrupted and no v19 checkpoint was found in the new session or mounted Version 36 output. Do not count it as completed. Version 37 is submitted as Save & Run All, using `scripts/kaggle_v0_19_entrypoint.py` synchronously. It mounts the saved 3Beethoven-v0.2 output and restores the hash-checked v15 adapter, installs the recorded training dependencies, then runs training, evaluation and output verification. Final Successful status is still pending. No scientific protocol changes; this is execution recovery.


## v19 完成後追加

正式 Version 37 Successful；第一輪觸發保留門檻。新概念 72→96/96，舊技能複核 59→56/96，恰有一個事件 10→1/12；不升級。所有待確認答案已複核，完整結果與備份見 [完成報告](STATS_V0_19_REPORT.md)。原預期與停止規則不回改。


## 已保存的固定權重分析 — Version 38

Version 38（347843399）Successful；分析輸出目錄 `3beethoven_v19_analysis`，656 筆回答；完整摘要、原始資料及補充語意複核見 [分析報告](STATS_V19_ANALYSIS_REPORT.md)。分析 ZIP 317,568 bytes，SHA-256 `5fdba3ca321bbc8408fa85c488fdbd19ee3482ff80dbc6a00c7bc19a5c04a3e2`，CRC 通過。

重跑 `scripts/kaggle_v19_analysis_entrypoint.py` 必須指定 **Version 37** 為來源輸入，因為兩個 adapter 與原始專案保存在 Version 37；Version 38 是分析輸出，不包含那兩個權重目錄。不要把 latest 自動解析成 Version 38 後直接重跑。Version 38 已完成兩模型保存權重的 SHA 核對及重新載入推論，並逐字重現各 24 筆原題。這不等於驗證 optimizer-state 訓練續跑。


</details>
