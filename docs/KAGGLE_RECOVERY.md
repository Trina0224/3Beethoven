# 最終成果恢復：final_clear（V58）

**已完成：最後採用學生 `final_clear`（Kaggle V58），直接從原 v15 接續訓練。** 同一份最終保留測試，v15 **107/144** → final_clear **144/144**；18 類全部不退步，9 類改善、9 類維持滿分。改正 37 題，保留原本正確 107 題，退步 0 題，待判 0 題。

## 已保存模型

[下載 V58 模型 ZIP](https://www.kaggle.com/code/trinashih/3beethoven-v0-2/output?scriptVersionId=348141342&select=3beethoven_final_clear_student.zip)（Kaggle Version 58，scriptVersionId `348141342`，Successful）。

- 檔名：`3beethoven_final_clear_student.zip`；97,086,130 bytes。
- ZIP 內模型路徑：`student/adapter`。
- ZIP SHA-256：`047873f5db94168bbf8a924dea41a260f7fdedcb2b344b1e5eeb6bb2c93f483f`。
- Adapter SHA-256：`60c5c7e956480484c8dacf5d5dbbf4bb17da6d46a49ecbff24e803b42621b2d5`。
- Base：`meta-llama/Llama-3.2-3B-Instruct`，revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`。

ZIP 已下載並驗證 CRC、檔案與權重雜湊。這是 LoRA adapter 加 tokenizer，載入時仍需上述 base；直接載入 final_clear adapter，不要再疊加 v15 adapter。GPU 已停止，沒有排程下一輪。

## 載入正確權重

1. 下載上述 V58 ZIP 並解壓；使用 `student/adapter`。
2. 取得已授權的 `meta-llama/Llama-3.2-3B-Instruct` base，固定 revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`；憑證放既有 secrets，不寫入程式庫。
3. 依封存的 `source/scripts/run_stats_final_clear.py` 使用相同量化、tokenizer、chat template 與 user prompt suffix，將 final_clear adapter 載入 base。不要先載入 v15 或 V57 adapter。
4. 要重現輸出，使用相同 clear prompt，greedy decoding、`max_new_tokens=160`。先查看封存評估檔內完整 prompt；不要以自行改寫的問法聲稱重現同一測試。

ZIP 是 adapter/tokenizer 而非完整 base。恢復模型不需 Run All 舊 notebook，不需重新訓練。原 v15 adapter SHA-256 為 `9369d52de4a886df9da0c872cd41bd4e01af0a38bf02ad724b5951c1a6b9f5d3`，只用於辨識本次起點。

## 封存內容

- `student/adapter/`：最終 LoRA 與 tokenizer。
- `student/training_complete.json`、`student/microbatches.jsonl`：訓練完成與 1,152 筆監督稽核。
- `evaluation/v15/`、`evaluation/student/`：development/test 原始與複核結果。
- `semantic_reviews.json`、`manual_reviews.json`、`pending_reviews.json`：等價性複核與空待判清單。
- `summary.json`、`RESULTS.md`：結果；`source/`：資料、協議、程式與結果來源。
- 各階段 `*_contract.json`：凍結比較條件。

可攜 ZIP 不含 optimizer checkpoint；Kaggle 完整輸出另保留最後 checkpoint。本次採固定最後權重，並未用最終測試挑選 checkpoint。

## 執行環境與已驗證範例

環境：torch 2.10.0+cu128、transformers 5.0.0、peft 0.19.1、bitsandbytes 0.50.2、accelerate 1.13.0、datasets 5.0.0。Base 使用 NF4 double quantization，fp16 計算；原始 adapter config 隨檔保存。

最終測試中，給定 E[X]=71、Var(X)=468、Y=4X+5，要求 E[Y²]，final_clear 輸出 `4**2*468+(4*71+5)**2`。完整原文與其餘例題見 [最終報告](STATS_FINAL_CLEAR_RESULTS.md) 及封存的 evaluation 檔。

本次 ZIP 已實際下載核對，144 題兩模型的 prompt、ID、題目雜湊及分數相符；GPU 已停止。GitHub 報告補記 V58 交付資訊，不改變已封存權重。
