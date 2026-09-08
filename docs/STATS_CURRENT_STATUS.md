# 目前狀態：V58 已撤回，受控重做仍在訓練前

截至本次文件修正，沒有可稱為全面優於原 v15 的新學生。

| 模型／工作 | 目前定位 |
|---|---|
| 原 v15 | 唯一可信的訓練起點與同題比較錨點；不是成功終點 |
| `final_clear`／Kaggle V58 | 歷史窄模板結果；採用與成功解讀已撤回 |
| 新的多樣教材 v2 | 已重建；720 new＋720 historical-train replay，73 項相關測試通過 |
| 新學生 | 尚未產生 |

## 為何撤回 V58

V58 在原先固定的 18 類 clear-interface 測試上確實得到 v15 **107/144**、V58 **144/144**，原報告中的逐題分數、權重及交付雜湊均保留。問題不在這些數字是否曾出現，而在它們只證明對狹窄模板分布的改善：教材與 train／development／test 共用過強的表述骨架，符號、分母、正負號、邊界、單位與支撐範圍等變化不足。因此原內部 gate 不能支持「全面比 v15 好」或「已成功蒸餾」的升級結論。

後續外部結構診斷只留下歷史聚合：

| 診斷（24 題） | 正確 |
|---|---:|
| 原 v15 | 10/24 |
| V58 | 16/24 |

這 24 題的逐題 raw bundle 未保存在 GitHub 或 Kaggle 的已保存版本。聚合只能支持撤回與重新設計教材的決定，不能在本輪充當可執行、可重現的選點或成功 gate，也不能用來改教材後再宣稱盲測。

## 現在正在做什麼

本輪維持原始目標：在有限、直接、清楚的統計列式範圍內，讓一個新學生全面勝過 v15；不把任務改成語文能力。

已完成的教材修正：

- 新教材為 train/development/final 每類 40/10/10；每個 contrast group 都按 family 逐組驗證，不再容許 binomial 或 interval 被 split 配額截斷；
- 720 筆歷史 train rows 已實際物化成 replay，與 720 筆新教材固定 1:1 交錯；retention development/final 不進訓練；
- train normalized template 為 48 種，單一模板最高 20 列；只採 givens-first/target-first 兩種直接問法，不增加語文任務；
- promotion 數學分固定只讀 math_correct＋executable；strict one-line 只作格式診斷。

進入老師與 GPU 前仍須完成並凍結：

- 18 類能力與結構軸的覆蓋矩陣，不以同模板換數字充數；
- 每題 prompt 與結構化語意的雙向核對；
- 獨立 oracle、合法等價式與已知錯式突變測試；
- train／development／final 以及歷史題的碰撞與洩漏稽核；
- v15、教材、老師回覆、grader 和執行設定的雜湊綁定；
- final test 在選點前不可見，以及失敗即停止、不得測後補跑的規則。

只有全部 preflight 通過，才會進入老師生成與一次 GPU 訓練。目前已有新教材與 replay，但沒有老師答案或模型改善。

[受控實驗協議](STATS_DIVERSE_EXPERIMENT_PROTOCOL.md) · [V58 撤回後記](STATS_V58_POSTMORTEM.md) · [下一輪狀態](STATS_NEXT_RUN_STATUS.md) · [執行交接](STATS_EXECUTION_HANDOFF.md) · [模型恢復](KAGGLE_RECOVERY.md)
