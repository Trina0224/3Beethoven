# 統計蒸餾整合比較：開跑方案草案

**審核後註記（2026-09-07）：本頁保留 GPT-5.6 原草案，不能依此直接開跑。現行規則與必修項目見 [開跑包審核](STATS_CONSOLIDATION_REVIEW.md)；49 步一輪上限、112 題選點及相對 v15 門檻取代下方原設定。**

更新：2026-09-07 02:25 PDT（America/Los_Angeles）。狀態：**工作包一完成，等待高階模型鎖定五項決策；教師與 GPU 均未啟動。**本頁是可審查草案，未定案前程式會主動拒絕付費呼叫及訓練。

## 要回答的問題

使用同一份完整教材及相同新增訓練預算，比較：

1. 原始 `meta-llama/Llama-3.2-3B-Instruct`＋全新 LoRA；
2. 原 v15 adapter 接續微調。

兩個起點都跑 seeds 2027、31415，共四次。這能檢驗目前訓練歷史是否限制整合；不能單獨證明教材順序、3B 容量或 QLoRA rank 的因果效果。

## 已準備的教材候選

| 項目 | 數量 | 用途 |
|---|---:|---|
| 故事 bundle | 96 | 先依 lineage 分成訓練 71、驗證 25，再產生教師請求 |
| 新目標 | 268 | 訓練 198、驗證 70；同故事包含相關所求量 |
| 第一批 pilot | 24 個訓練故事／65 個目標 | 先檢查教師格式、數學正確率、拒收原因與實際用量 |
| 固定歷史教師 replay | 192 | 與通過核對的新教師訓練目標混合；不使用輔助類別冒充八類 replay |
| 預期訓練列數 | 最多約 390 | 198 個新目標全數通過時的上限；實際數量依核對後接受率 |

涵蓋歷史八類、四類組合、四事件及平均數／變異數／二階動差的同故事對照。每筆保留 `story_id`、`lineage_id`、concept、split、來源、問題 hash、表述與容易混淆的所求量。具體新測試不存在於教師請求檔；測試的故事、參數、措辭家族及答案都必須由獨立 builder 產生。

## 教師產生與限額

- 固定教師：`meta-llama/llama-3.3-70b-instruct`；temperature 0；每故事最多 400 output tokens；最多兩次嘗試。
- 教師只收到問題與 JSON 回覆格式，不收到參考算式、bindings、答案或測試內容。
- 每次回覆保存原文、實際模型／provider cache、usage、cost、逐題判定及拒收原因。教師算式必須可執行且通過已凍結結構核對；數值相同但結構未核實者不自動收入。
- 建議限額：pilot 最多 48 calls／US$0.50；完整 96 故事累計最多 192 calls／US$3.00。任何已完成回覆缺少 cost 時，下一次呼叫停止。這些是風險上限，仍待高階模型批准，不是預估必定花滿。
- pilot 若出現系統性題意／格式問題，保存最小證據後交回；不得靠放寬 grader 提高接受率。pilot 合格標準尚待決策。

## 訓練與線上選點

沿用 v20 的 4-bit NF4 QLoRA、LoRA r=16／alpha=32／dropout=0.05、LR `2e-5` constant、effective batch 8 及 768-token 截斷保護。四次使用同一組已接受資料；相同 seed 共用故事群組順序。原生起點建立新 LoRA，v15 起點強制核對 adapter SHA-256 `9369d52de4a886df9da0c872cd41bd4e01af0a38bf02ad724b5951c1a6b9f5d3`。

- 建議最多 48 updates，每 12 updates 做一次線上驗證；資料不足一輪 48 updates 時，以一次完整曝光的可用更新數為上限，不循環補滿。
- 每次選點用固定 91 題：舊八類 32、組合四類 24、四事件 16、新表述 19。它們全來自 validation split；完整測試不參與選點。
- 第一次所有逐類門檻及新表述總門檻同時通過，即保存並停止。到上限仍未通過就記為失敗，不加第二輪、第五個 seed 或額外 updates。
- 四個 selected／diagnostic checkpoint 都完成後，才執行新的 288 題測試及永久 MC 240 responses。MC 若未達 promotion gate，不回頭改選 checkpoint。

最壞情況為四個 run×四次線上驗證×91 responses＝1,456 個生成答案；完整評估每個模型 528 responses。原始模型與原 v15 基準可各快取一次，四個訓練候選各一次，共最多 3,168 個完整評估 responses。GPU 訓練上限為 4×48＝192 updates；本案的時間風險主要在生成式驗證與語義複核，需由 pilot 實測速度後再估算，不先承諾完成時數。

## 評分與保存

- `math_correct` 與 `executable` 分開保存；不補上模型漏掉的 `/60`、變異數項或事件分支。
- 已測試：正確參考式、乘除重組的等價秒／分鐘換算、漏 `/60`、漏二階動差變異數項、把 exactly_one 寫成 same，以及不安全語法。
- 每個 run 隔離目錄，保存 contract、資料／順序 hash、checkpoint adapter、raw answers、逐題判定、log、實際曝光數與 SHA-256。每個 run 完成即打包；二進位權重保存於 Kaggle，GitHub 保存程式、資料 manifest、結果與下載資訊。
- 只有驗證選出的 checkpoint 進入獨立測試；沒有合格點時，最後一次只標示 diagnostic。最終表格逐類呈現兩起點×兩 seed，不以總分遮住局部失效。

## 需要高階模型鎖定的五項決策

1. 是否批准 96 個故事／268 個新目標及 192 筆 replay 的比例；是否先以 24 故事 pilot 作品質 gate。
2. 是否批准 pilot US$0.50／48 calls、完整累計 US$3／192 calls 的硬上限，以及 pilot 最低接受率／系統性錯誤停止條件。
3. 是否批准最多 48 updates、每 12 updates 驗證及「一次曝光、不循環補步數」。接受率確定後需用實際 token 長度再核一次。
4. 是否批准[決策草案](STATS_CONSOLIDATION_DECISION_DRAFT.json)中逐 suite／category 的最低率、新表述 75% 門檻及永久 MC 122/240；這些數值目前只是具體提案。
5. 是否批准[獨立測試藍圖](STATS_CONSOLIDATION_HOLDOUT_BLUEPRINT.json)，以及是否另花教師 quota 在預先指定的 holdout 子集測教師，讓文章能直接比較 vanilla／學生／教師。

批准時把決策檔 `protocol_status`、`validation_gate.status` 及 selection-validation matrix 的 `status` 改為 `frozen_for_execution`，同步寫入最終數值與 hash。教師腳本與訓練腳本才會解除拒絕執行。
