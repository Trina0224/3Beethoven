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
