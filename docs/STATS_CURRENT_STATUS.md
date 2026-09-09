# 目前狀態：最新 step 180 學生通過原始 3B final blind 比較

狀態：**蒸餾目標成功；正式評測完成。**

2026-09-08 PDT，使用先前封存且未參與訓練或 checkpoint 選擇的 180 題 diverse final blind，完成原始 3B 與最新 step 180 學生的同條件比較。

| 指標 | 原始 3B | step 180 |
|---|---:|---:|
| 正確 | 45/180 | **170/180** |
| Strict one-line expression | 26/180 | **180/180** |
| Pending | 66 | **0** |

配對轉移：127 題錯轉對、2 題對轉錯、43 題共同答對、8 題共同答錯。18 類中 17 類提高、1 類持平、0 類下降。學生剩餘 10 題均已人工檢查，屬實際數學或代入錯誤，沒有 grader 誤殺。

## 現行模型

- Base：`meta-llama/Llama-3.2-3B-Instruct`
- Base revision：`0cb88a4f764b7a12671c53f0838cd831a0843b95`
- Student：原始 base 直接掛 step 180 LoRA；不可再疊加 v15。
- Adapter SHA-256：`ff89a22f097e4db457dab287042ae36520ec6b9b36f26e5ed11fe42337c04f23`
- 權重：[Kaggle Version 63](https://www.kaggle.com/code/trinashih/3beethoven-v0-2/output?scriptVersionId=348384546)，`3Beethoven_latest_step180_weights.zip`

## 歷史協議邊界

原 v15 仍是本輪訓練來源與歷史 selection 錨點。凍結 selection protocol 的 `no_checkpoint_passed` 結論不變，因為 development 上的逐題零損失硬門檻未達成，且存在已記錄的執行偏差。後續 final blind 是使用者明確改定自然比較對象後進行的正式產品評估，因此可以宣稱限定範圍內的蒸餾成功，但不能宣稱舊 frozen protocol 已追溯通過。

[正式結果](STATS_DIVERSE_FINAL_BLIND_RESULTS.md) · [機器摘要](STATS_DIVERSE_FINAL_BLIND_RESULTS.json) · [訓練歷史](STATS_DIVERSE_RUN_RESULTS.md) · [執行交接](STATS_EXECUTION_HANDOFF.md)
