# Kaggle 恢復：原 v15 比較錨點與 V58 歷史封存

## 現行起點：原 v15

新的受控實驗只可從原 v15 恢復，不可從 V58 或 V55–V57 接續。v15 是目前唯一可信的受控起點與比較錨點，**不是成功終點**。

- Kaggle 原始保存：Version 29，Successful。
- 後續已驗證可恢復位置：[Kaggle Version 42](https://www.kaggle.com/code/trinashih/3beethoven-v0-2/output?scriptVersionId=347852579)，Successful。
- Adapter 路徑：`3beethoven_stats_v0_15/adapter/adapter_model.safetensors`。
- Adapter SHA-256：`9369d52de4a886df9da0c872cd41bd4e01af0a38bf02ad724b5951c1a6b9f5d3`。
- 原始 ZIP：`3beethoven_stats_v0_15.zip`，93,892,056 bytes。
- ZIP SHA-256：`49b1e411edab050f0aae694b0b1b80181613bca5afacc8d79dd75abfe0eed097`。
- Base：`meta-llama/Llama-3.2-3B-Instruct`，revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`。

恢復時先鎖定 base revision，再載入 v15 LoRA adapter；不得先疊加其他 adapter。憑證使用 Kaggle secrets，不寫入程式庫。任何權重路徑或 hash 不符都應在訓練前停止。

## v15 證據界線

v15 在當時同一批 64 題中相對 v14 由 30/64 提升到 44/64，14 題錯轉對、0 題對轉錯；舊 MC 為 127/240。但當時二階動差與仿射 Poisson 變異數仍各 0/8。因此它適合作為累積歷史最可信的起點，不代表它已全面成功。

後來外部 24 題結構診斷只留有 v15 10/24 的聚合；逐題 raw bundle 未保存在 GitHub 或 Kaggle 已保存版本，不能拿它作本輪可重現基準 gate。

## V58：只供歷史稽核

[V58 歷史 ZIP](https://www.kaggle.com/code/trinashih/3beethoven-v0-2/output?scriptVersionId=348141342&select=3beethoven_final_clear_student.zip)仍保留，Kaggle Version 58／scriptVersionId `348141342` 顯示 Successful；`Successful` 只表示 notebook 執行與保存成功，不表示模型已通過現行採用標準。

- ZIP：`3beethoven_final_clear_student.zip`，97,086,130 bytes。
- ZIP SHA-256：`047873f5db94168bbf8a924dea41a260f7fdedcb2b344b1e5eeb6bb2c93f483f`。
- Adapter 路徑：`student/adapter`。
- Adapter SHA-256：`60c5c7e956480484c8dacf5d5dbbf4bb17da6d46a49ecbff24e803b42621b2d5`。

V58 在原窄模板測試的 107/144 → 144/144 結果仍可從封存資料稽核；其採用與成功解讀已撤回。不要把 V58 adapter 當新一輪父模型，也不要把其 144/144 當跨分布能力證據。

[目前狀態](STATS_CURRENT_STATUS.md) · [V58 撤回後記](STATS_V58_POSTMORTEM.md) · [V58 歷史結果](STATS_FINAL_CLEAR_RESULTS.md) · [v15 備份清單](MODEL_BACKUP_STATUS.json)
