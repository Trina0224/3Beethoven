# Colab 重跑：原始 3B vs 最新 step180 的 diverse final blind 考試

## 最新權重：Hugging Face

正式模型頁：[kozakurayuki/3Beethoven-step180](https://huggingface.co/kozakurayuki/3Beethoven-step180)。權重與 `adapter_config.json` 已上傳；遠端 LFS SHA-256 與完成 170/180 評測的 step 180 權重一致。下方 Kaggle ZIP 保留為歷史備份。

- HF revision：`682d5555b0e6115135ac7b9b5d718d2abef186de`
- Adapter SHA-256：`ff89a22f097e4db457dab287042ae36520ec6b9b36f26e5ed11fe42337c04f23`
- [直接下載 adapter_model.safetensors](https://huggingface.co/kozakurayuki/3Beethoven-step180/resolve/682d5555b0e6115135ac7b9b5d718d2abef186de/adapter_model.safetensors?download=true)

這是 LoRA adapter，載入時仍需固定 revision 的原始 Llama 3.2 3B Instruct；不要疊加 v15。Tokenizer 使用原始 base。


正式評測已於 2026-09-08 PDT 在 Colab 完成。原始 `meta-llama/Llama-3.2-3B-Instruct` 為 **45/180**，最新 step180 LoRA adapter 為 **170/180**。這份交接供完全相同條件重跑。

## 這次比較回答什麼

- baseline：原始 3B base，不掛任何 LoRA。
- candidate：重新載入原始 3B base，再直接掛最新 step180 LoRA。
- 考卷：`docs/STATS_DIVERSE_FINAL_BLIND.json`，180 題，18 類各 10 題。
- 主分：`primary_correct`，沿用 diverse run 的 math/executable grader contract。
- v15 不會被載入，也不是這次比較的 baseline。

最新 step180 adapter 身分：

- Adapter SHA-256：`ff89a22f097e4db457dab287042ae36520ec6b9b36f26e5ed11fe42337c04f23`
- ZIP output 頁面：https://www.kaggle.com/code/trinashih/3beethoven-v0-2/output?scriptVersionId=348384546
- ZIP 檔名：`3Beethoven_latest_step180_weights.zip`
- ZIP SHA-256：`4a97c1070fd81ff2fb570699eb630043562b9b4968b799dfed6dfb410fdf036e`

## Colab 步驟

請使用 GPU runtime。T4 足夠跑 4-bit inference。

```bash
pip install -q "transformers>=4.45" "peft>=0.12" "bitsandbytes" "accelerate" "safetensors" "datasets"
git clone https://github.com/Trina0224/3Beethoven.git
```

把 `3Beethoven_latest_step180_weights.zip` 放到 `/content/`，再執行：

```bash
cd /content
unzip -q 3Beethoven_latest_step180_weights.zip -d /content/3beethoven_step180_zip
export HF_TOKEN="<your Hugging Face token>"
python /content/3Beethoven/scripts/evaluate_stats_diverse_final_blind_colab.py \
  --adapter /content/3beethoven_step180_zip \
  --final-blind /content/3Beethoven/docs/STATS_DIVERSE_FINAL_BLIND.json \
  --output /content/3beethoven_final_blind_eval
```

如果要先用很快的小考確認環境正常：

```bash
python /content/3Beethoven/scripts/evaluate_stats_diverse_final_blind_colab.py \
  --adapter /content/3beethoven_step180_zip \
  --final-blind /content/3Beethoven/docs/STATS_DIVERSE_FINAL_BLIND.json \
  --output /content/3beethoven_final_blind_eval_smoke \
  --limit 6
```

## 輸出檔案

- `/content/3beethoven_final_blind_eval/original_3b_base.json`
- `/content/3beethoven_final_blind_eval/latest_step180_adapter.json`
- `/content/3beethoven_final_blind_eval/comparison_summary.json`
- `/content/3beethoven_final_blind_eval/contract.json`

最後一行 console 會以 `FINAL_BLIND_COMPARISON` 開頭，欄位如下：

```json
{
  "original_3b_base_correct": 0,
  "latest_step180_correct": 0,
  "transitions": {},
  "paired_losses_vs_original_3b_base": 0,
  "paired_gains_vs_original_3b_base": 0
}
```

成功重跑的正式輸出應為原始 3B `45/180`、step180 `170/180`；配對 transitions 為 `0->0: 8`、`0->1: 127`、`1->0: 2`、`1->1: 43`。

Evaluator 固定從 base model revision 載入 tokenizer。不要改成優先讀 adapter ZIP 內的 `tokenizer_config.json`；舊檔指定的 `TokenizersBackend` 在目前 Colab Transformers 會造成 `Tokenizer class TokenizersBackend does not exist`。

若要把結果下載回來：

```bash
cd /content
zip -r final_blind_eval_results.zip 3beethoven_final_blind_eval
```

## 從 Hugging Face 直接準備 Colab 權重

已有 HF_TOKEN 環境變數時，可取代上傳與解壓縮 Kaggle ZIP 的步驟：

```python
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="kozakurayuki/3Beethoven-step180",
    revision="682d5555b0e6115135ac7b9b5d718d2abef186de",
    local_dir="/content/3beethoven_step180_zip/adapter",
    allow_patterns=["adapter_model.safetensors", "adapter_config.json"],
    token=os.environ.get("HF_TOKEN"),
)
```

保留與 Transformers 4.57.6 相容的 `huggingface_hub>=0.34,<1.0`；不要為下載權重無限制升級套件。
