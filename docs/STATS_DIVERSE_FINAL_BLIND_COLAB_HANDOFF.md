# Colab 交接：原始 3B vs 最新 step180 的 diverse final blind 考試

Kaggle 目前因 GPU quota 超過而不能完成 final blind 比較。這份交接只做缺少的那個比較：原始 `meta-llama/Llama-3.2-3B-Instruct` 對最新 step180 LoRA adapter，考卷使用已封存的 diverse final blind。

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

上面的 0 只是 schema 範例，不是預期分數。

若要把結果下載回來：

```bash
cd /content
zip -r final_blind_eval_results.zip 3beethoven_final_blind_eval
```

