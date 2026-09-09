# Diverse final blind 正式結果：蒸餾目標成功

狀態：**完成。最新 step 180 學生在原始 3B 對照下通過專案目標。**

2026-09-08 PDT，依使用者最終指定，以未掛 LoRA 的原始 `meta-llama/Llama-3.2-3B-Instruct` 作自然 baseline，與同一個 base 直接掛載最新 step 180 LoRA 的學生，在先前封存且未用於訓練或選點的 180 題 final blind 上比較。兩者使用相同 tokenizer、prompt、greedy decoding、grader 與 4-bit 載入條件；沒有載入或疊加 v15。

## 正式成績

| 指標 | 原始 3B | 最新 step 180 |
|---|---:|---:|
| 自動正確 | 45/180（25.0%） | **170/180（94.4%）** |
| Strict one-line expression | 26/180 | **180/180** |
| Pending | 66 | **0** |

配對轉移為 `0→0: 8`、`0→1: 127`、`1→0: 2`、`1→1: 43`，即 127 題由錯轉對、2 題由對轉錯，淨增加 125 題。即使把原始 3B 的全部 66 題 pending 都暫時判對，其上限也只有 111/180，仍低於學生的 170/180。

## 各類別

| 類別（各 10 題） | 原始 3B | step 180 |
|---|---:|---:|
| at_least_one | 4 | 10 |
| binomial | 3 | 9 |
| both | 10 | 10 |
| exactly_one | 0 | 10 |
| interval | 0 | 9 |
| moment_mean | 4 | 10 |
| moment_second | 0 | 8 |
| moment_variance | 0 | 9 |
| neither | 7 | 9 |
| poisson_scaled | 0 | 10 |
| poisson_second | 1 | 10 |
| poisson_variance | 5 | 10 |
| process_scaled | 1 | 7 |
| process_second | 0 | 10 |
| process_variance | 1 | 10 |
| same | 0 | 10 |
| uniform_conditional | 0 | 9 |
| uniform_mean | 9 | 10 |

18 類中 17 類提高、1 類持平、0 類下降。這不是由少數題型灌高總分。

## 剩餘錯誤檢查

學生的 10 題失敗均為實際列式或代入錯誤，沒有 grader 誤殺，也沒有 pending：

- `interval_001`：區間半寬把端點相加，應為相減。
- `process_scaled_003`、`004`、`005`：漏乘 Poisson rate 或觀察時間。
- `moment_second_004`、`006`：二階動差展開錯誤。
- `moment_variance_006`：把平均數平方項誤加進變異數。
- `neither_007`：漏掉第一個事件的補集；這是兩題 paired loss 之一。
- `uniform_conditional_008`：條件截斷端點代入錯誤。
- `binomial_009`：`r=n` 時仍多乘一次 `(1-p)`；這是另一題 paired loss。

弱點最集中在 `process_scaled`（7/10）與 `moment_second`（8/10）；這些是日後修訂教材時最明確的局部方向，不影響本次限定範圍內的成功判定。

## 結論邊界

本結果支持的主張是：response distillation 使原始 3B 在約定的直接統計列式範圍，由 25.0% 提升至 94.4%，並學會 180/180 的嚴格一行算式輸出。它不主張一般語文能力、未知統計領域能力或完整蒸餾因果消融。

先前以 v15 作選點錨點的 frozen protocol 仍保持原始歷史結論 `no_checkpoint_passed`；本次後續 final blind 比較不能倒推改寫當時的 protocol compliance。使用者之後明確指定自然產品比較應為「原始 3B vs 最新學生」，所以目前專案成果採用 step 180 作最新成功學生，而 v15 保留為歷史訓練起點與研究錨點。

## 可重現身份與證據

- Base：`meta-llama/Llama-3.2-3B-Instruct`
- Base revision：`0cb88a4f764b7a12671c53f0838cd831a0843b95`
- step 180 adapter SHA-256：`ff89a22f097e4db457dab287042ae36520ec6b9b36f26e5ed11fe42337c04f23`
- Final blind split SHA-256：`9b216083cafab17f76b0c28f3e0941e9727234ede0d2d270056b64c5c2d6524a`
- Grader fingerprint：`f73c1d38cbd5e1ab9454275300907aa6c6a43c3833a303441ffb1311c18ad68b`
- 完整結果 ZIP：[`final_blind_eval_results.zip`](../final_blind_eval_results.zip)，83,251 bytes，SHA-256 `4f3e57aeb129a6d43870496841fdf1b806e688ac9e8bfced12b0c5b968cbad54`
- 機器可讀摘要：[STATS_DIVERSE_FINAL_BLIND_RESULTS.json](STATS_DIVERSE_FINAL_BLIND_RESULTS.json)
- 最新權重：[Kaggle Version 63](https://www.kaggle.com/code/trinashih/3beethoven-v0-2/output?scriptVersionId=348384546)，下載 `3Beethoven_latest_step180_weights.zip`
