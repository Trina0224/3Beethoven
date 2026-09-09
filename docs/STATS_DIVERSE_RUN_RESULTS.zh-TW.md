# 多樣教材單輪訓練與 selection 歷史結果

> 後續狀態：使用者另行指定以原始 3B 作自然 baseline，並授權解封 final blind。step 180 在該正式評測得到 **170/180**，原始 3B 為 **45/180**，因此限定統計列式蒸餾目標成功。見 [正式 final blind 結果](STATS_DIVERSE_FINAL_BLIND_RESULTS.md)。以下保留當時 v15 frozen selection 的原始結論，不追溯改寫。

狀態：**`no_checkpoint_passed`；保留原 v15，不解封 final blind。**

本輪確實完成一個新學生的唯一訓練軌跡。它在固定 180 題 new development 上大幅提高 aggregate 分數，legacy development 的 aggregate 也沒有下降；但凍結協議要求保留 v15 原本答對的每一題，step 180 仍有兩題 paired loss。因此這是可觀察的能力改善，不是「全面比 v15 好」的畢業結果。

## 模型與訓練身份

- 起點：原 v15；adapter SHA-256 `9369d52de4a886df9da0c872cd41bd4e01af0a38bf02ad724b5951c1a6b9f5d3`。
- 訓練序列：1,440 rows，由 720 筆新教材與 720 筆歷史 train replay 以 1:1 組成。
- 訓練長度：180 optimizer updates。
- 最後記錄 loss：約 `0.20015`。
- step 180 adapter SHA-256：`ff89a22f097e4db457dab287042ae36520ec6b9b36f26e5ed11fe42337c04f23`。

## 固定 development 與 retention 結果

主分沿用凍結定義：只有 `math_correct is True and executable is True` 才計分；格式診斷不併入數學主分。

| 模型／checkpoint | New development／180 | New 各類不退 | New paired losses | Unresolved pending | Legacy development／72 | Legacy gate |
|---|---:|---|---:|---:|---:|---|
| 原 v15 | 91（另有 5 題 pending） | 比較錨點 | — | — | 51 | 比較錨點 |
| step 60 | 139 | 是 | 2 | 2 | 69 | 各類不退；0 loss；0 pending |
| step 120 | 167 | 是 | 0 | 1 | 72 | 各類不退；0 loss；0 pending |
| step 180 | 自動 172；暫行 173 | 是 | 2 | 0（唯一放行後） | 72 | 各類不退；0 loss；0 pending |

各 checkpoint 的 new development 失敗項目為：

- step 60 paired losses：`diverse_development_moment_variance_006`、`diverse_development_moment_second_007`；unresolved pending：`diverse_development_process_scaled_002`、`diverse_development_moment_second_007`。
- step 120 paired losses：無；unresolved pending：`diverse_development_process_scaled_002`。
- step 180 唯一 pending 經下列精確放行後 unresolved pending 為 0；paired losses：`diverse_development_moment_variance_006`、`diverse_development_binomial_009`。

step 180 的暫行 173/180 只包含使用者明示、唯一一次的人工放行：

- ID：`diverse_development_process_scaled_001`
- raw：`Expression: ((12/19)**2)*(10**2)`
- raw SHA-256：`327520af57a1a3eb0098b0cd9ff40592aa9cec561976a44427bf0ba05df3fb37`
- 精確值：`14400/361`

這一題在本結果中可作 provisional capability credit，但它是在看到輸出後才人工定案，不把自動 pending 改寫掉，也不算符合凍結協議的 `pending=0` 要求。

step 180 在 new development 的 18 類 aggregate 均不低於 v15，但仍失去 v15 原本答對的兩題：

1. `diverse_development_moment_variance_006`
2. `diverse_development_binomial_009`

因此，即使接受唯一的 pending 暫行放行，paired-loss gate 仍是 `2 != 0`，選點結果固定為 `no_checkpoint_passed`。本輪沒有選中 checkpoint；legacy final 依協議只對 selected checkpoint 執行，所以是**按協議未跑，不是忘記驗證**。final blind 也沒有讀取或解封。

## 結論邊界

**Empirical capability：** step 180 在固定 new development 由 v15 的 91/180 提高至自動 172/180，或含唯一暫行放行的 173/180；legacy development aggregate 為 72/72。這足以記錄本輪模型出現顯著、廣泛的可觀察改善；本輪沒有消融對照，不把改善歸因到單一教材或 replay 成分。

**Frozen-protocol compliance：** 不合格。除 step 180 的兩題 paired loss 已直接違反零損失硬門檻外，本輪 v15 baseline 是訓練後補做，step 180 也早於固定的 60→120→180 選點順序被查看，唯一人工放行則是輸出後定案。這些 execution deviations 都必須保留，不能把本輪重命名為完整遵循凍結協議的成功實驗。

原 v15 仍是保留中的比較錨點與可恢復起點。step 180 可保留供研究與稽核，但不取代 v15、不稱為畢業學生，也不以未解封的 final blind 推測成績。

## 保存

Kaggle Quick Save 已成功完成：Version 62，scriptVersionId `348379527`。保存頁面：[3beethoven-v0-2 Version 62](https://www.kaggle.com/code/trinashih/3beethoven-v0-2?scriptVersionId=348379527)。

本輪評估固定以 grader fingerprint `f73c1d38cbd5e1ab9454275300907aa6c6a43c3833a303441ffb1311c18ad68b` 執行。New development 的 rows/split 與 order SHA-256 分別為 `28ad5fc66e2f99ab137d340b03c4b5db6616ff89200bb0e787af80fe63f364ce`、`5290523a8ea6a62dc2ce6b4d8f17b1485bb108467c2fe3d6e0872542a9fdb8d8`；legacy development 的 rows 與 order SHA-256 分別為 `52bf46dc3efe2a69966be57c0573e14dad68c52b832b3d31c76b646fc5f4c54c`、`c183c543341a56e16e96705a825734d3bd637a2a2031b0dcb750be457364207c`。

機器可讀摘要見 [STATS_DIVERSE_RUN_RESULTS.json](STATS_DIVERSE_RUN_RESULTS.json)。現行門檻原文保留於 [STATS_DIVERSE_EXPERIMENT_PROTOCOL.md](STATS_DIVERSE_EXPERIMENT_PROTOCOL.md)，不因本輪輸出事後改寫。
