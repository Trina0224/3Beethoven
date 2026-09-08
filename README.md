# 3Beethoven

## 目前狀態：V58 已撤回，受控重做仍在訓練前

`final_clear`（Kaggle V58）不再是採用學生，也不能稱為本專案的成功結果。它在原先固定、狹窄的 18 類模板測試上確實得到原 v15 **107/144 → 144/144**；這些數字保留為歷史結果，但後續檢查發現教材和評估的結構多樣性不足，原 gate 無法支持「全面比 v15 好」的結論。

一組後來做的外部結構診斷只有歷史聚合：v15 **10/24**、V58 **16/24**。逐題 raw bundle 未保存在 GitHub 或 Kaggle 的已保存版本，因此它只能解釋為何撤回，**不能成為新一輪可執行或可重現的 gate**。

| 模型 | 目前定位 |
|---|---|
| **原 v15** | 唯一可信的受控起點與比較錨點；不是成功終點 |
| **final_clear / V58** | 歷史窄模板結果；升級與採用資格已撤回 |
| V55–V57 等 | 歷史診斷學生；均未證明全面優於 v15 |
| 新學生 | **尚未產生** |

目前先完成教材、prompt↔語意、oracle、grader、資料洩漏、錯式突變、雜湊綁定與最終測試隔離等訓練前檢查，再允許老師生成或 GPU 訓練。範圍仍是直接、清楚的統計列式，不把目標改成語文能力。只規劃一個學生、一次訓練；是否成功必須由凍結後的同題比較和逐類零退步決定，不能靠內部模板滿分或訓練 loss 宣稱。

## 可恢復的比較錨點

原 v15 adapter：

- 路徑：`3beethoven_stats_v0_15/adapter/adapter_model.safetensors`
- SHA-256：`9369d52de4a886df9da0c872cd41bd4e01af0a38bf02ad724b5951c1a6b9f5d3`
- ZIP：`3beethoven_stats_v0_15.zip`，93,892,056 bytes
- ZIP SHA-256：`49b1e411edab050f0aae694b0b1b80181613bca5afacc8d79dd75abfe0eed097`
- Base：`meta-llama/Llama-3.2-3B-Instruct`，revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`

V58 權重仍可作歷史稽核，不應載入為新一輪父模型。其原始窄模板結果與交付雜湊見 [撤回後記](docs/STATS_V58_POSTMORTEM.md) 及 [歷史結果報告](docs/STATS_FINAL_CLEAR_RESULTS.md)。

## 文件入口

- [目前狀態](docs/STATS_CURRENT_STATUS.md) · [下一輪狀態](docs/STATS_NEXT_RUN_STATUS.md)
- [受控實驗協議](docs/STATS_DIVERSE_EXPERIMENT_PROTOCOL.md) · [執行交接](docs/STATS_EXECUTION_HANDOFF.md)
- [V58 撤回後記](docs/STATS_V58_POSTMORTEM.md) · [V58 歷史結果 JSON](docs/STATS_FINAL_CLEAR_RESULTS.json)
- [v15／歷史模型恢復](docs/KAGGLE_RECOVERY.md) · [完整研究流程](docs/STATS_V01_V19_RESEARCH_FLOW.md)

原始音樂概念與早期實驗保留為歷史背景。Kaggle 版本號是 notebook 保存版本，不能直接當作學生世代或品質排名；模型身份應以 adapter 路徑與雜湊判定。
