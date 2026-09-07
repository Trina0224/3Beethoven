# 補充：原 v15 數值準備對照

2026-09-07 PDT，補充推論前登記。低 LR 修復 seed 2027 已完成且未過門檻，31415 尚在既定一遍內。程式碼顯示 baseline 分支未呼叫 prepare_model_for_kbit_training，student 分支有呼叫；先前 336/336 重現只驗證各自原條件，不能隔離此差異。

在兩個既定 repair seed 完成後，另外以原 v15 不變權重、相同底模 revision、同環境及相同 112 道選點題，加入 prepare_model_for_kbit_training 且 is_trainable=True 後做 greedy 推論。這是事前追加 112 次唯讀診斷，明確將原控制推論上限由 552 擴至 664；訓練仍至多 66 updates、教師新呼叫 0、整段 GPU 90 分鐘上限不變。此補充不增加 seed、不選擇新 checkpoint、不用控制結果事後改原 gate。

假設：若相同原 v15 在 prepared 條件的原答／分數改變，baseline 與 student 存在數值配置混雜，需要下一輪統一配置後另凍結基準；不能把全部差異算成訓練效果。若 112 原答相同，則這個具體混雜未在現有題集顯現。此對照仍不能證明 resume 與不中斷訓練數值等價。
