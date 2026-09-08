# 目前狀態：final_clear 已成功完成

**已完成：最後採用學生 `final_clear`（Kaggle V58），直接從原 v15 接續訓練。** 同一份最終保留測試，v15 **107/144** → final_clear **144/144**；18 類全部不退步，9 類改善、9 類維持滿分。改正 37 題，保留原本正確 107 題，退步 0 題，待判 0 題。

| 題型（每類 8 題） | 原 v15 | final_clear | 變化 |
|---|---:|---:|---:|
| 至少一個發生 | 7/8 | 8/8 | +1 |
| 二項分布指定成功次數 | 7/8 | 8/8 | +1 |
| 兩者都發生 | 8/8 | 8/8 | +0 |
| 恰好一個發生 | 0/8 | 8/8 | +8 |
| 樣本量增加後區間上界 | 4/8 | 8/8 | +4 |
| 仿射變換後 E[Y] | 8/8 | 8/8 | +0 |
| 仿射變換後 E[Y²] | 4/8 | 8/8 | +4 |
| 仿射變換後 Var(Y) | 8/8 | 8/8 | +0 |
| 兩者都不發生 | 3/8 | 8/8 | +5 |
| Poisson 仿射變換後 Var(Y) | 8/8 | 8/8 | +0 |
| Poisson E[X²] | 7/8 | 8/8 | +1 |
| Poisson Var(X) | 8/8 | 8/8 | +0 |
| Poisson 過程仿射變換 Var(Y) | 3/8 | 8/8 | +5 |
| Poisson 過程計數 E[X²] | 0/8 | 8/8 | +8 |
| Poisson 過程計數 Var(X) | 8/8 | 8/8 | +0 |
| 兩者結果相同 | 8/8 | 8/8 | +0 |
| 條件均勻分布總等待 E[T] | 8/8 | 8/8 | +0 |
| 均勻分布 E[T] | 8/8 | 8/8 | +0 |
| **合計** | **107/144** | **144/144** | **+37** |

Development：v15 51/72 → final_clear 72/72。共 21 筆等價性複核，原始輸出與判定依據保留；待判 0。最終兩模型逐題比較為 107 題皆對、37 題由錯轉對，沒有退步。

本輪使用固定、清楚的數學問法，學生輸出代入數值的算式，算術由程式處理。共 18 類，訓練 1,152 筆、development 72 題、final test 144 題；各集合語意鍵與完整數值題情境不重疊。教材由助理根據核實的數學規則修正、擴充，其中 17 類有既有 70B 老師答案作依據，均勻分布均值類為助理規則教材；不是 1,152 筆新生成的 70B 回答。本輪新增雲端老師 API 呼叫為 0。

事前門檻：至少 +5/144、各類不退步、無待判。只訓練一次，固定最後權重，不用測試挑模型。成功範圍為這 18 類清楚列式題型。

## 已保存模型

[下載 V58 模型 ZIP](https://www.kaggle.com/code/trinashih/3beethoven-v0-2/output?scriptVersionId=348141342&select=3beethoven_final_clear_student.zip)（Kaggle Version 58，scriptVersionId `348141342`，Successful）。

- 檔名：`3beethoven_final_clear_student.zip`；97,086,130 bytes。
- ZIP 內模型路徑：`student/adapter`。
- ZIP SHA-256：`047873f5db94168bbf8a924dea41a260f7fdedcb2b344b1e5eeb6bb2c93f483f`。
- Adapter SHA-256：`60c5c7e956480484c8dacf5d5dbbf4bb17da6d46a49ecbff24e803b42621b2d5`。
- Base：`meta-llama/Llama-3.2-3B-Instruct`，revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`。

ZIP 已下載並驗證 CRC、檔案與權重雜湊。這是 LoRA adapter 加 tokenizer，載入時仍需上述 base；直接載入 final_clear adapter，不要再疊加 v15 adapter。GPU 已停止，沒有排程下一輪。

[完整報告](STATS_FINAL_CLEAR_RESULTS.md) · [結果 JSON](STATS_FINAL_CLEAR_RESULTS.json) · [事前協議](STATS_FINAL_CLEAR_PROTOCOL.md) · [恢復指南](KAGGLE_RECOVERY.md)
