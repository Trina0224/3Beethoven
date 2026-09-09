# 目前狀態：多樣教材學生已完成，沒有 checkpoint 通過畢業門檻

狀態：**`no_checkpoint_passed`。保留原 v15；final blind 未解封。**

| 模型／工作 | 目前定位 |
|---|---|
| 原 v15 | 受控起點與比較錨點；adapter SHA-256 `9369d52de4a886df9da0c872cd41bd4e01af0a38bf02ad724b5951c1a6b9f5d3` |
| `final_clear`／Kaggle V58 | 歷史窄模板結果；採用與成功解讀仍為撤回狀態 |
| 多樣教材 v2 | 已完成 720 new＋720 historical-train replay 的唯一訓練軌跡 |
| step 180 新學生 | Aggregate 最佳，但因兩題 paired loss 未畢業、不取代 v15 |

## 本輪實際完成內容

- 訓練資料：1,440 rows，包含 720 筆新教材與 720 筆歷史 train replay。
- 訓練長度：180 optimizer updates；最後記錄 loss 約 `0.20015`。
- step 180 adapter SHA-256：`ff89a22f097e4db457dab287042ae36520ec6b9b36f26e5ed11fe42337c04f23`。
- Kaggle Quick Save：已成功保存為 [Version 62](https://www.kaggle.com/code/trinashih/3beethoven-v0-2?scriptVersionId=348379527)，scriptVersionId `348379527`。

固定驗收結果如下；主分只使用 `math_correct is True and executable is True`：

| 模型／checkpoint | New development／180 | New paired losses | Unresolved pending | Legacy development／72 |
|---|---:|---:|---:|---:|
| 原 v15 | 91（另有 5 題 pending） | — | — | 51 |
| step 60 | 139 | 2 | 2 | 69 |
| step 120 | 167 | 0 | 1 | 72 |
| step 180 | 自動 172；暫行 173 | 2 | 0（唯一放行後） | 72 |

三個 checkpoint 在 new 與 legacy development 的逐類分數均不低於 v15；legacy development 也都是 0 paired loss、0 pending。step 60 因 new development 的 paired loss 與 pending 失敗；step 120 因一題 unresolved pending 失敗；step 180 則在唯一 pending 精確放行後，仍因兩題 paired loss 失敗。逐題 ID 見[完整結果](STATS_DIVERSE_RUN_RESULTS.md)。

step 180 的暫行 173 只接受使用者明示的唯一 pending：`diverse_development_process_scaled_001`。原始輸出為 `Expression: ((12/19)**2)*(10**2)`，raw SHA-256 為 `327520af57a1a3eb0098b0cd9ff40592aa9cec561976a44427bf0ba05df3fb37`，精確值為 `14400/361`。這個 credit 只用於 provisional capability 解讀，不覆寫原自動 pending，也不算 frozen-protocol compliance。

step 180 在 new development 的 18 類 aggregate 都不低於 v15，但仍有兩題 paired loss：

1. `diverse_development_moment_variance_006`
2. `diverse_development_binomial_009`

凍結門檻要求 paired losses 為 0。因此 step 60、120、180 均未通過全部選點 gate；沒有 selected checkpoint。Legacy final 依協議只對 selected checkpoint 執行，所以本輪是按協議未跑、不是忘記驗證；final blind 亦未讀取或解封。

## 能成立與不能成立的結論

可以成立：本輪在有限、直接的統計列式範圍內出現顯著且廣泛的 empirical capability 改善；new development 從 v15 的 91/180 提高到 step 180 的自動 172/180，legacy development aggregate 亦提高到 72/72。

不能成立：全面比 v15 好、畢業、取代 v15，或完整遵循 frozen protocol。除了兩題 paired loss，本輪 baseline 是訓練後補做，step 180 早於固定 60→120→180 選點順序被查看，唯一人工放行也在看到輸出後才定案。這些 execution deviations 必須原樣保留。

[完整結果](STATS_DIVERSE_RUN_RESULTS.md) · [機器可讀摘要](STATS_DIVERSE_RUN_RESULTS.json) · [受控實驗協議](STATS_DIVERSE_EXPERIMENT_PROTOCOL.md) · [下一輪狀態](STATS_NEXT_RUN_STATUS.md) · [模型恢復](KAGGLE_RECOVERY.md)
