# 下一輪狀態：教材 v2 已重建，仍未呼叫老師或 GPU

V58 已撤回採用資格；原 v15 恢復為唯一可信的起點與比較錨點。v15 並不是成功終點，新學生尚未產生。

本輪的順序固定為：先完成教材與評估 preflight，留下可重現收據；再生成／驗證老師教材；只有所有 gate 通過才啟動一次學生訓練。不得先訓練再補教材理由，不得查看 final test 後修題、調參或重跑。

目前邊界：

- 目標限於直接、清楚的統計列式，不考一般語文能力；
- 一個 v15 起點學生、一條訓練軌跡、一次訓練；
- 教材須覆蓋負數、零、分數、分母變化、邊界、單位與非零支撐範圍等結構差異；
- 新教材 720 筆與實際歷史 train replay 720 筆固定一新一舊；只做 retention 事後評分不算 replay；
- 每個 contrast group 必須完整，且每類同時使用兩種受控直接問法；
- 選點只用 development 與既有回歸集；final test 保持隔離；
- 成功必須總分增加、逐類不退步、v15 原本正確題零損失、待判為零；
- 任一必要 preflight 或資料完整性檢查失敗就停止，不花老師或 GPU 成本補救。

歷史 24 題外部診斷的 v15 10/24、V58 16/24 沒有逐題 raw bundle 可供 GitHub／Kaggle 重現，只用於解釋 V58 撤回，不列入本輪執行 gate。

[現行協議](STATS_DIVERSE_EXPERIMENT_PROTOCOL.md) · [目前狀態](STATS_CURRENT_STATUS.md) · [執行交接](STATS_EXECUTION_HANDOFF.md)
