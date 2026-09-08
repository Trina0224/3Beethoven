# 最後一次實驗：final_clear 在約定列式範圍內勝過原 v15

完成時間：2026-09-07T23:05:46.898008-07:00（America/Los_Angeles）

## 最終結論

**成功學生是 `final_clear`，不是 V57 repeat_control。** 最後同題保留測試：原 v15 **107/144（74.3%）**，final_clear **144/144（100%）**。18 類全部不退步；9 類改善、9 類維持滿分。原先正確的 107 題全部保住，改正 37 題，退步 0 題。所有待判答案均完成複核。

這次符合使用者的成功定義：在事前約定的統計列式範圍內，整體優於 v15 並保留各類能力。先前 V57 的事件局部 16/16 不符合此定義，已撤回「最後成功學生」定位，僅保留歷史實驗紀錄。

## 固定範圍與成功門檻

[執行前協議](STATS_FINAL_CLEAR_PROTOCOL.md) 在新教材生成、基準推論與訓練前提交（commit `08f2afc6ed0426e55cd31826df07ae3bf67d1b6b`）。成功須新測試總分至少 +5/144，且 18 類逐類正確數均不低於 v15，無待判答案；實際 +37，全部通過。

採直接、固定的數學問法，明列 E[X]、Var(X)、E[Y]、Var(Y)、E[Y²] 或事件與條件；學生輸出已代入數值的算式，算術交給程式。不要求陌生措辭、隱含定義、選擇題或自由形式語文泛化。原 v15 和新學生的題目、提示與評分方式完全相同。

每類 64 筆訓練、4 題 development、8 題 final test，共 1,152／72／144。訓練和測試共用明確格式、數值不同；掃描 3,149 個歷史語意 key 排除重複任務，train/development/test 的語意 key 與故事參數不跨分割。144 題 final test 皆為不同參數故事，從未用來訓練或選 checkpoint。這是內部生成、固定介面的新數值測試，不宣稱任意自然語言或所有統計問題均已解決。

## 完整逐類結果

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

開發檢查另為原 v15 51/72、final_clear 72/72。它不替代最後測試，也未用來修改資料、超參數或挑 checkpoint。

## 具體改正的算式

### 恰好一個發生

A and B are independent events. P(A)=37/100; P(B)=49/100. Find the probability that exactly one occurs.

- 原 v15：`Expression: (37 / 100) * (49 / 100) + (37 / 100) * (1 - 49 / 100) + (1 - 37 / 100) * (49 / 100)`
- final_clear：`Expression: (37/100)*(1-(49/100))+(1-(37/100))*(49/100)`

### 仿射變換後 E[Y²]

E[X]=71; Var(X)=468. Define Y=4*X+5. Find E[Y**2].

- 原 v15：`Expression: 4 ** 2 * 71 ** 2 + (4 * 71 + 5) ** 2`
- final_clear：`Expression: 4**2*468+(4*71+5)**2`

### Poisson 過程仿射變換 Var(Y)

A homogeneous Poisson process has rate 59 arrivals per minute. X counts arrivals during 11 minutes. Define Y=3*X+15. Find Var(Y).

- 原 v15：`Expression: 3 ** 2 * 59`
- final_clear：`Expression: 3**2*(59*(11))`

### Poisson 過程計數 E[X²]

A homogeneous Poisson process has rate 45 arrivals per minute. X counts arrivals during 224 seconds. Find E[X**2].

- 原 v15：`Expression: 45 * (224 / 60) * (224 / 60 + 45 / 60)`
- final_clear：`Expression: (45*(224/60))+(45*(224/60))**2`

### 二項分布指定成功次數

X is the number of successes in 5 independent trials, each with success probability 41/100. Find P(X=2).

- 原 v15：`Expression: (41 / 100) ** 2 * (1 - 41 / 100) ** (5 - 2) / 2`
- final_clear：`Expression: comb(5,2)*(41/100)**2*(1-41/100)**(5-2)`

### 樣本量增加後區間上界

A normal-theory confidence interval is [48, 175]. Sample size is multiplied by 49; center, confidence level and population standard deviation stay fixed. Find the new upper endpoint.

- 原 v15：`Expression: (48 + (175 - 48) / (2 * 7)) * (1 / (2 / 7)) + (175 - 48) / (2 * 7)`
- final_clear：`Expression: (48+175)/2+(175-48)/(2*7)`

## 教材來源與這次改變

保留已有老師列式作規則依據，以明確、統一的數學欄位和標準算式，展開每類 64 個新數值練習，均衡涵蓋原有題型與組合概念。每筆參考列式都以獨立數學 oracle 和精確計算器交叉核對，所有訓練參考也通過既有列式評分器。並非單純增加改寫數量。

**新數值教材是助手編寫的參數展開，不是新呼叫 70B 所得的原始回答。** 17 類記錄了既有已驗證老師答案作規則來源；uniform_mean 沒有在所用來源集合找到對應老師錨點，明確標記為助手提供的均勻分布規則。這是承接老師蒸餾學生的 teacher-informed SFT／response-distillation 後續實驗，含明示的人工式教材修正與合成展開，不是純未修改老師輸出或 logits 蒸餾。新老師 API 呼叫為 0。

這次一起改變了提示標準化、例題分布、參數練習和學習率，結果不能單獨歸因於其中一項。低訓練 loss 沒有被用來宣稱成功，成功依據是事前固定的同題保留測試。

## 學生身份、訓練與成本範圍

- 起點：**原 v15**，不是 v14，也不是 V55／V56／V57。
- 原 v15 SHA-256：`9369d52de4a886df9da0c872cd41bd4e01af0a38bf02ad724b5951c1a6b9f5d3`。
- 最後 adapter SHA-256：`60c5c7e956480484c8dacf5d5dbbf4bb17da6d46a49ecbff24e803b42621b2d5`。
- Base：`meta-llama/Llama-3.2-3B-Instruct`，revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`。
- seed 2027；一輪、144 updates；batch 1、gradient accumulation 8、LR 2e-5 constant；沿用原 LoRA config、NF4 double quantization、fp16 compute。
- 實際 1,152 筆 microbatches 的 input／label hash 與順序全部核對；16,431 supervised tokens；392 個 adapter tensors 均為有限值。
- 訓練 loss 0.035813239695724204；訓練約 10.72 分鐘；基準、訓練與全部推論合計 24.37 分鐘（不含前置連線與最後文件整理）。
- 只有一個新學生、一次訓練，固定最後一步；沒有追加 seed、epoch、模型混合或根據測試挑 checkpoint。
- 共 432 次生成：兩個模型各 72 development＋144 test。0 次新老師 API 呼叫，不等於整體服務或 GPU 成本為零。

## 評分證據與重現

保留原始 `development.json`／`test.json`，另寫 `*_reviewed.json`；原始回答沒有修補。共 21 筆等價式複核，含原 v15 的 Poisson `mu*(mu+1)` 形式，正確者完整加回。新學生的 144/144 不靠補寫缺項或供給參考答案。

[完整結果摘要](STATS_FINAL_CLEAR_RESULTS.json) · [固定資料 manifest](STATS_FINAL_CLEAR_MANIFEST.json) · [壓縮固定教材及題集](STATS_FINAL_CLEAR_DATA.json.gz.b64) · [資料程式](../scripts/stats_final_clear.py) · [實際訓練／推論程式](../scripts/run_stats_final_clear.py) · [複核及打包程式](../scripts/finish_stats_final_clear.py)

固定資料提交：`e0ba8d0d17bbcadf1f392d9cfc56cc1a29b612e2`。原始回答、來源、評分、microbatch 紀錄與最後權重一起放在 `3beethoven_final_clear_student.zip`。最後 adapter 位於 **`student/adapter`**；此資料夾是 LoRA adapter，須搭配上述固定 base，不能當成獨立完整 base 權重，也不要再疊加 v15 adapter。模型下載入口見專案 README 與 KAGGLE_RECOVERY。

驗證環境：torch 2.10.0+cu128、transformers 5.0.0、peft 0.19.1、bitsandbytes 0.50.2、accelerate 1.13.0、datasets 5.0.0。推論採 greedy、max_new_tokens=160，使用相同 Llama chat template 與既有數值前處理。

**本輪工作完成後停止，不再自動安排下一個學生。**

## 已保存模型

[下載 V58 模型 ZIP](https://www.kaggle.com/code/trinashih/3beethoven-v0-2/output?scriptVersionId=348141342&select=3beethoven_final_clear_student.zip)（Kaggle Version 58，scriptVersionId `348141342`，Successful）。

- 檔名：`3beethoven_final_clear_student.zip`；97,086,130 bytes。
- ZIP 內模型路徑：`student/adapter`。
- ZIP SHA-256：`047873f5db94168bbf8a924dea41a260f7fdedcb2b344b1e5eeb6bb2c93f483f`。
- Adapter SHA-256：`60c5c7e956480484c8dacf5d5dbbf4bb17da6d46a49ecbff24e803b42621b2d5`。
- Base：`meta-llama/Llama-3.2-3B-Instruct`，revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`。

ZIP 已下載並驗證 CRC、檔案與權重雜湊。這是 LoRA adapter 加 tokenizer，載入時仍需上述 base；直接載入 final_clear adapter，不要再疊加 v15 adapter。GPU 已停止，沒有排程下一輪。

交付資訊於模型封存後補入 GitHub；沒有因此重新訓練或改動封存模型。
