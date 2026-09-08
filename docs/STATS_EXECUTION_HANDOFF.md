# 完成交接：final_clear / V58

**已完成：最後採用學生 `final_clear`（Kaggle V58），直接從原 v15 接續訓練。** 同一份最終保留測試，v15 **107/144** → final_clear **144/144**；18 類全部不退步，9 類改善、9 類維持滿分。改正 37 題，保留原本正確 107 題，退步 0 題，待判 0 題。

## 交接決定

- 本次最後一輪已完成；無下一輪，GPU 已停止。
- 起點是原 v15，不是 v14 或 V55–V57。final_clear 是延續後完整 LoRA adapter，不需再疊加 v15。
- 約定範圍是 18 類固定清楚的統計列式題型；不加語文改寫測驗。
- 1,152 筆、1 epoch、144 updates、seed 2027、LR 2e-5；固定最終權重。沒有重跑或從多名新學生挑結果。
- 訓練約 10.72 分鐘；基準、訓練及 432 次推論約 24.37 分鐘。新增雲端老師 API 呼叫 0。

本輪使用固定、清楚的數學問法，學生輸出代入數值的算式，算術由程式處理。共 18 類，訓練 1,152 筆、development 72 題、final test 144 題；各集合語意鍵與完整數值題情境不重疊。教材由助理根據核實的數學規則修正、擴充，其中 17 類有既有 70B 老師答案作依據，均勻分布均值類為助理規則教材；不是 1,152 筆新生成的 70B 回答。本輪新增雲端老師 API 呼叫為 0。

## 已保存模型

[下載 V58 模型 ZIP](https://www.kaggle.com/code/trinashih/3beethoven-v0-2/output?scriptVersionId=348141342&select=3beethoven_final_clear_student.zip)（Kaggle Version 58，scriptVersionId `348141342`，Successful）。

- 檔名：`3beethoven_final_clear_student.zip`；97,086,130 bytes。
- ZIP 內模型路徑：`student/adapter`。
- ZIP SHA-256：`047873f5db94168bbf8a924dea41a260f7fdedcb2b344b1e5eeb6bb2c93f483f`。
- Adapter SHA-256：`60c5c7e956480484c8dacf5d5dbbf4bb17da6d46a49ecbff24e803b42621b2d5`。
- Base：`meta-llama/Llama-3.2-3B-Instruct`，revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`。

ZIP 已下載並驗證 CRC、檔案與權重雜湊。這是 LoRA adapter 加 tokenizer，載入時仍需上述 base；直接載入 final_clear adapter，不要再疊加 v15 adapter。GPU 已停止，沒有排程下一輪。

## 證據與操作

[完整報告](STATS_FINAL_CLEAR_RESULTS.md)、[JSON](STATS_FINAL_CLEAR_RESULTS.json)、[事前協議](STATS_FINAL_CLEAR_PROTOCOL.md)、[資料來源清單](STATS_FINAL_CLEAR_MANIFEST.json)、[模型恢復](KAGGLE_RECOVERY.md)。ZIP 包含原始與複核輸出、逐題比較依據、資料與程式。以同一批題目比較模型，不把歷史異質分數當成同一量尺。

GitHub 是目前文件來源；本機舊 checkout 可能有其他未提交工作。不要建立分支／PR、改寫歷史、加入共同作者或 blanket staging；必要文件應逐檔精準更新。不要依舊工作包再訓練。
