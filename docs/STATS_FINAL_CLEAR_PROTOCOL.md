# 最後一次：清楚列式範圍下優於原 v15

使用者在看到 V57 事件局部改善、舊題退步後，明確否定把局部成功當作全面升級，並授權最後一次認真實驗。本協議在新教材生成、新基準推論和新訓練前固定。原 v15 仍是保留學生，V57 不算成功的替代模型。之前文件的最後成功命名不再適用。

## 任務與假設

固定直接問法、明確已知量與 E/Var 所求量，只輸出代入數值的算式；不增加陌生改寫、隱含定義、選擇題或自主計算要求。假設：教師列式規則的均衡教材、同格式新數值練習能改善弱項，同時保留原 v15 在同一清楚介面下的能力。這是整體教材配方實驗，不隔離單一原因。

18 類：五種獨立雙事件（both/neither/exactly_one/at_least_one/same）；仿射變換後 mean/variance/second_moment；Poisson count variance、scaled variance、second moment；Poisson process count variance、scaled variance、second moment；uniform mean、conditional total mean；binomial exact count；normal interval upper endpoint under increased sample size。涵蓋原八類及既有組合概念，不將弱項剔除。

## 教材與分割

每類 64 train、4 development、8 final test，共 1,152/72/144。固定數字生成 seed 580058；先分割再執行。train/dev/test 的故事參數與語意 task key 不重疊；掃描歷史教材與評估的 key 以排除新測試撞題。固定直接問法跨 split 共用，這是明示任務內的新數值測試，不是語文泛化測試。

每類保留既有已驗證老師資料作規則來源；將老師正確列式規則以參數模板展開新數值，獨立數學 oracle 驗算。來源與模板展開必須明確標示為 teacher-derived、assistant-authored parameter augmentation，不能稱為新 70B 原始回答。沿用使用者允許直接修正老師教材的授權；0 新雲端老師呼叫。若某類找不到既有正確老師規則，不默默宣稱其為老師輸出。

## 一個起點、一輪、一次選擇

原 v15 SHA256 9369d52de4a886df9da0c872cd41bd4e01af0a38bf02ad724b5951c1a6b9f5d3；基礎 Llama-3.2-3B-Instruct revision 0cb88a4f764b7a12671c53f0838cd831a0843b95。不用 v14/V55/V56/V57 作起點。

先量 v15 development 72 題以確認固定介面下的表現；不以結果改課程、門檻或超參數。v15 final test 與新學生 final test 在訓練之後同場評估，test 不用選點。若 development 已全對，也完整報告 ceiling，不把 baseline 能力說成訓練成果。

seed 2027，1 epoch=144 optimizer updates，batch 1、accumulation 8、LR 2e-5、constant schedule、fp16、原 v15 LoRA config、NF4 double quantization。均衡打亂 1,152 rows；prompt masking 與 max length 768 保持既有驗證流程。固定最後一步；最多保存一個中途恢復點（72）及最後 adapter，不依分數選 checkpoint，不追加 epoch、seed、混合權重或第二個学生。整個 GPU 工作硬上限 110 分鐘，失敗時保存可用證據並停止。

## 評分與成功

同題、同提示、相同 greedy/max_new_tokens=160，比較原 v15 和最終學生。沿用既有數學／可執行列式評分器與恆等式複核；不要求唯一字串、不補學生漏掉的項。相等數值但無列式證據不能自動過關；所有 pending 在最後判定前一併複核。

**成功必須 final test 總分至少比 v15 多 5/144，而且 18 個類別每類正確數均不低於 v15，無未決評分。** development 僅呈現診斷，不取代此門檻。保留每一題配對結果及退步題。未達門檻即明說最後一次未成功、保留 v15，不挑局部結果命名成功。

## 交付

一份結果文件含資料來源、同場逐類與配對結果、訓練與推論紀錄、最後 adapter 定位及成本；一份 Kaggle 可下載模型與執行來源。更新 README/目前狀態，撤回 V57 全面成功的錯誤定位。完成後關閉 GPU，不再新增實驗。
