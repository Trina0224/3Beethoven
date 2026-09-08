# 3Beethoven

**已完成：最後採用學生 `final_clear`（Kaggle V58），直接從原 v15 接續訓練。** 同一份最終保留測試，v15 **107/144** → final_clear **144/144**；18 類全部不退步，9 類改善、9 類維持滿分。改正 37 題，保留原本正確 107 題，退步 0 題，待判 0 題。

這項結論限於約定的 18 類清楚統計列式介面；不擴張為任意語文表述或所有統計問題的能力聲明。

| 模型 | 目前定位 |
|---|---|
| **final_clear / Kaggle V58** | **最後採用的成功學生** |
| 原 v15 | 本輪起點與同題比較基準 |
| V57 repeat_control 等 | 歷史實驗；先前「最後成功學生」定位已撤回 |

| 同題評估 | 原 v15 | final_clear |
|---|---:|---:|
| Development | 51/72 | 72/72 |
| Final test | 107/144 | 144/144 |
| 恰好一個發生 | 0/8 | 8/8 |
| Poisson 過程 E[X²] | 0/8 | 8/8 |
| 仿射變換 E[Y²] | 4/8 | 8/8 |

## 已保存模型

[下載 V58 模型 ZIP](https://www.kaggle.com/code/trinashih/3beethoven-v0-2/output?scriptVersionId=348141342&select=3beethoven_final_clear_student.zip)（Kaggle Version 58，scriptVersionId `348141342`，Successful）。

- 檔名：`3beethoven_final_clear_student.zip`；97,086,130 bytes。
- ZIP 內模型路徑：`student/adapter`。
- ZIP SHA-256：`047873f5db94168bbf8a924dea41a260f7fdedcb2b344b1e5eeb6bb2c93f483f`。
- Adapter SHA-256：`60c5c7e956480484c8dacf5d5dbbf4bb17da6d46a49ecbff24e803b42621b2d5`。
- Base：`meta-llama/Llama-3.2-3B-Instruct`，revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`。

ZIP 已下載並驗證 CRC、檔案與權重雜湊。這是 LoRA adapter 加 tokenizer，載入時仍需上述 base；直接載入 final_clear adapter，不要再疊加 v15 adapter。GPU 已停止，沒有排程下一輪。

## 本次工作

本輪使用固定、清楚的數學問法，學生輸出代入數值的算式，算術由程式處理。共 18 類，訓練 1,152 筆、development 72 題、final test 144 題；各集合語意鍵與完整數值題情境不重疊。教材由助理根據核實的數學規則修正、擴充，其中 17 類有既有 70B 老師答案作依據，均勻分布均值類為助理規則教材；不是 1,152 筆新生成的 70B 回答。本輪新增雲端老師 API 呼叫為 0。

只跑一次、1 epoch、144 updates，固定採用最後權重。訓練約 10.72 分鐘，基準、訓練及全部 432 次推論合計約 24.37 分鐘（不含準備與封存）。最終測試未用於選 checkpoint、修改教材或訓練。事前成功門檻為總分至少 +5/144、18 類各不低於 v15、無待判；實際 +37，通過。

## 文件入口

- [完整結果、18 類比較與錯誤改正範例](docs/STATS_FINAL_CLEAR_RESULTS.md)
- [機器可讀結果](docs/STATS_FINAL_CLEAR_RESULTS.json)
- [目前狀態](docs/STATS_CURRENT_STATUS.md) · [模型恢復](docs/KAGGLE_RECOVERY.md)
- [事前協議](docs/STATS_FINAL_CLEAR_PROTOCOL.md) · [資料清單與來源](docs/STATS_FINAL_CLEAR_MANIFEST.json)
- [固定單輪執行程式](scripts/run_stats_final_clear.py) · [專案規格](PROJECT_SPEC.md)
- [完成交接](docs/STATS_EXECUTION_HANDOFF.md) · [後續狀態：無下一輪](docs/STATS_NEXT_RUN_STATUS.md)
- [V57 歷史實驗](docs/STATS_SEMANTIC_COURSE_RESULTS.md) · [舊局部範例](docs/STATS_BOUNDED_SUCCESS_EXAMPLE.md)

原始音樂概念與早期實驗保留為歷史背景。Kaggle 版本號是 notebook 保存版本，不能直接當作學生世代或品質排名；模型身份應以 adapter 路徑與雜湊判定。
