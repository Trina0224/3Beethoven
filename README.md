# 3Beethoven

## 目前狀態：新學生已完成，但沒有 checkpoint 通過畢業門檻

多樣教材 v2 的唯一學生訓練與 development／legacy development 選點已完成。step 180 在固定 new development 由原 v15 的 **91/180** 提高至自動 **172/180**；依使用者明示暫行放行唯一 pending 後為 **173/180**，legacy development 為 **72/72**。但它仍失去 v15 原本答對的兩題，因此 paired-loss gate 未過，結果是 **`no_checkpoint_passed`**。本輪不解封 final blind，原 v15 繼續保留。

這個結論分成兩層：本輪使用新教材與 replay 後，確實觀察到顯著的 empirical capability 改善；但 paired losses、訓練後才補跑 baseline，以及輸出後人工放行 pending，都使它不能宣稱符合 frozen protocol，更不能稱為全面勝過 v15 的畢業學生。完整數字與 execution deviations 見[本輪結果](docs/STATS_DIVERSE_RUN_RESULTS.md)。

| 模型 | 目前定位 |
|---|---|
| **原 v15** | 唯一可信的受控起點與比較錨點；不是成功終點 |
| **final_clear / V58** | 歷史窄模板結果；升級與採用資格已撤回 |
| V55–V57 等 | 歷史診斷學生；均未證明全面優於 v15 |
| 多樣教材 step 180 | Aggregate 大幅改善；因 2 題 paired loss 未畢業、不取代 v15 |

本輪使用 720 筆新 train 與 720 筆物化歷史 train replay，固定 1:1 組成 1,440-row 訓練序列，共 180 optimizer updates，最後記錄 loss 約 `0.20015`。step 60／120／180 的 new development 分別為 139／167／172；legacy development 分別為 69／72／72。step 180 adapter SHA-256 為 `ff89a22f097e4db457dab287042ae36520ec6b9b36f26e5ed11fe42337c04f23`。

結果與 notebook 已成功保存為 [Kaggle Version 62](https://www.kaggle.com/code/trinashih/3beethoven-v0-2?scriptVersionId=348379527)，scriptVersionId `348379527`。

## 可恢復的比較錨點

原 v15 adapter：

- 路徑：`3beethoven_stats_v0_15/adapter/adapter_model.safetensors`
- SHA-256：`9369d52de4a886df9da0c872cd41bd4e01af0a38bf02ad724b5951c1a6b9f5d3`
- ZIP：`3beethoven_stats_v0_15.zip`，93,892,056 bytes
- ZIP SHA-256：`49b1e411edab050f0aae694b0b1b80181613bca5afacc8d79dd75abfe0eed097`
- Base：`meta-llama/Llama-3.2-3B-Instruct`，revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`

V58 權重仍可作歷史稽核，不應載入為新一輪父模型。其原始窄模板結果與交付雜湊見 [撤回後記](docs/STATS_V58_POSTMORTEM.md) 及 [歷史結果報告](docs/STATS_FINAL_CLEAR_RESULTS.md)。

## 文件入口

- [本輪結果](docs/STATS_DIVERSE_RUN_RESULTS.md) · [機器可讀摘要](docs/STATS_DIVERSE_RUN_RESULTS.json)
- [目前狀態](docs/STATS_CURRENT_STATUS.md) · [下一輪狀態](docs/STATS_NEXT_RUN_STATUS.md)
- [受控實驗協議](docs/STATS_DIVERSE_EXPERIMENT_PROTOCOL.md) · [執行交接](docs/STATS_EXECUTION_HANDOFF.md)
- [多樣教材協議](docs/STATS_DIVERSE_CURRICULUM_PROTOCOL.md) · [實際 replay rows](docs/STATS_DIVERSE_REPLAY.json)
- [V58 撤回後記](docs/STATS_V58_POSTMORTEM.md) · [V58 歷史結果 JSON](docs/STATS_FINAL_CLEAR_RESULTS.json)
- [v15／歷史模型恢復](docs/KAGGLE_RECOVERY.md) · [完整研究流程](docs/STATS_V01_V19_RESEARCH_FLOW.md)

原始音樂概念與早期實驗保留為歷史背景。Kaggle 版本號是 notebook 保存版本，不能直接當作學生世代或品質排名；模型身份應以 adapter 路徑與雜湊判定。
