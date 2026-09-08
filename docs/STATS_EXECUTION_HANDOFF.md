# 執行交接：受控重做尚未進入訓練

## 決定

- 撤回 `final_clear`／V58 的採用與成功定位；保留其 107/144 → 144/144 窄模板結果作歷史證據。
- 只從原 v15 開始；不得從 V55–V58 或其他偏科候選接續。
- v15 是比較錨點，不是已達標終點。
- 只建立一個新學生、執行一次訓練；目前尚無新學生或新成績。
- 任務仍是直接統計列式，算術由程式處理；不加入一般語文能力或無邊界自然語言泛化。

## 執行前硬鎖

在老師呼叫或載入 GPU 訓練套件前，必須逐項通過並保存收據：

1. v15 adapter SHA-256 為 `9369d52de4a886df9da0c872cd41bd4e01af0a38bf02ad724b5951c1a6b9f5d3`；base revision 為 `0cb88a4f764b7a12671c53f0838cd831a0843b95`。
2. 教材、development、final 的題目數、類別數、結構軸覆蓋與雜湊符合現行協議。
3. prompt 中的數值、條件、單位、事件與結構化 semantics 一致；oracle 與 grader 的正反例／錯式突變測試全過。
4. train／development／final 以及歷史題之間沒有完整題、語意鍵或參數情境碰撞。
5. STATS_DIVERSE_REPLAY.json 必須含 720 筆來源為歷史 train 的完整 rows；training plan 必須證明 720 new＋720 replay 嚴格 1:1 交錯，development/test 不得出現。
6. final test 在 checkpoint 選定與人工等價式定案前不可載入；測後不得調整或再跑。
7. 老師結果須綁定請求與題目雜湊，保留原始回覆、usage、cost、parse 與 rejection；缺成本或超預算即停止。

只有以上收據齊全，才能把準備狀態改成「允許訓練」。目前不要根據歷史 notebook 的舊工作包直接 Run All。

## 評估界線

V58 後來的外部結構診斷只有聚合 v15 10/24、V58 16/24；逐題 raw bundle 未保存在 GitHub 或 Kaggle 保存版。因此它不是本輪可執行 gate，也不可拿 16/24 當新學生要追的盲測分數。新的 success gate 必須使用事前凍結、完整保存的同題證據，且逐類零退步、配對損失為零、待判為零。

[受控協議](STATS_DIVERSE_EXPERIMENT_PROTOCOL.md) · [目前狀態](STATS_CURRENT_STATUS.md) · [V58 撤回後記](STATS_V58_POSTMORTEM.md) · [恢復錨點](KAGGLE_RECOVERY.md)
