# 多 seed 訓練：執行前協定

使用者已授權一口氣完成不同 seed 訓練、分析及保存。兩組各三次，共六次；不重新呼叫老師。

- v15：固定原 v14 adapter，固定保存的 200 train／30 validation 老師教材，原三 epochs、LR 2e-5、cosine、warmup 5、batch 8，依 validation loss 選 checkpoint。
- v19：固定原 v15 adapter，固定 480 筆教材，原最多四 epochs、LR 2e-5、constant、batch 8、原每輪保留與 plateau 停止規則。未過保留門檻仍評估診斷 checkpoint，明確區分保留的原 v15。
- 各組 seed 1919、2027、31415；六次均在本輪環境重新訓練，歷史單次另列，不併入平均。v15 歷史 seed 是 1515。
- 所有來源權重先核對 SHA-256。執行 manifest 在第一輪訓練前保存套件、GPU、來源與資料 hashes。固定同一基底 revision 與量化方式；種子影響訓練隨機狀態及資料順序。不宣稱 CUDA bitwise determinism。
- 每 run 都評估永久 MC 240 題、舊列式 96 題、新鏈 96 題、固定擾動 88 題。原 runner 額外的歷史測試／驗證照留；完全相同的結果可直接複用，不能用不同 prompt 替換。
- 停止與選點沿用凍結自動 grader，語意複核不回溯更改訓練停止。最後報自動與複核分數；保存每題 raw、各輪 adapter 與 trainer checkpoint。
- 報每 run、三次平均、樣本標準差與範圍；family、改寫與 MC 旋轉群組另列。不挑最好 seed，不根據本輪測試修改後續 run。三次變異只作初步估計，不能當作已知精確噪音底線。
- 若執行遇到環境或保存故障，只修執行問題；不得改教材、學習率或門檻。未完成的 run 不當作零分或忽略。無預先追加第 4 個 seed，本輪以六次完整記錄收尾。

預期（結果前）：若總分三題差確實小於訓練波動，三 seed 結果可能跨越原差距；若 exactly_one 失效跨 seed 重現，表示不是單次偶發。若 seed 改變後原模板仍高分、改寫仍低分，支持教材／目標辨識的系統性限制。這些都是待驗證假說。


執行修正：Version 39 在訓練前因來源仍掛載 Version 38 而停止，沒有更新。來源盤點確認 Version 29 同時包含原 v14、v15 adapter 及 v15 的 train/validation examples。改直接掛載 Version 29，以提交的程式與凍結題目補上後續版本來源；不變更訓練設定。

正式輸入補充：Version 40 也在訓練前停止；確認動態掛載只改了互動 session，正式輸入仍指向舊版本。改將 Version 29 的兩個父模型與 v15 教材、已提交後續程式／凍結題目保存為完整準備輸出，再由短入口 `kaggle_seed_replication_formal.py` 執行。這是保存／掛載修正，不是新增模型版本或結果導向調參。

推論一致性補充（Version 42 開始載入模型、尚未看到任何 seed 評分時記錄）：原 v15 runner 的 MC 評估經 `prepare_model_for_kbit_training` 載入，v19 固定權重評估不經該步驟。此步驟可能改變非量化張量 dtype。三次 v15 內部比較仍一致，但跨組永久 MC 要在六次訓練完成後，以相同直接 PeftModel 推論載入方式補跑三份 v15 MC；保留原 runner MC，另列一致載入的錨點結果。這不涉及重新訓練、種子選擇或修改評分規則。


## 事後 review 提出的下次規則（2026-09-06 23:23 PDT）

下一次採保守停止規則：第一輪若所有 family 通過事前凍結的保留門檻，立即保存並停止更新，不再等待兩輪停滯或四輪上限。這不代表每類滿分，也不等於自動升級；仍需獨立固定測試和原定成功條件。此規則尚待前瞻驗證，僅追加紀錄，不回改上面的本次原始規則。[review 與證據界線](STATS_REVIEW_HANDOFF.md)。
