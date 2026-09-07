# 第一批學生訓練結果

2026-09-07。兩個 seed 均完成 33 updates、epoch 1.0；所有待判定項已複核。兩者均未出現全 family 過關的 checkpoint，保存第 33 步診斷 adapter，不取代 v15。

本批是事前另立 protocol 的 **filtered-pilot 診斷批次**：原 v15 起跑，67 筆逐題驗證的新教材＋192 筆固定 replay，LR 2e-5。完整教師教材 gate 仍未過；不是原定 198 筆新教材／fresh-base 與 v15 的四次完整比較。

| 模型／seed | 步數 | 舊技能 /48 | 組合题 /48 | 事件题 /16 | 全門檻 |
|---|---:|---:|---:|---:|---|
| 原 v15，同環境 | 0 | 34 | 38 | 12 | 基準 |
| 2027 | 12 | 24 | 37 | 15 | 未過 |
| 2027 | 24 | 23 | 38 | 15 | 未過 |
| 2027 | 33 | 29 | 39 | 14 | 未過 |
| 31415 | 12 | 28 | 40 | 15 | 未過 |
| 31415 | 24 | 35 | 41 | 11 | 未過 |
| 31415 | 33 | 22 | 41 | 11 | 未過 |

觀察與界線：

- 2027 最後的組合與事件門檻過關，但舊 exactly_one 由基準 5/6 變 3/6、poisson_scaled 由 3/6 變 1/6，未保住舊技能。
- 31415 的組合 scaled_variance 達 12/12（最深層 4/4），但舊 interval 由 6/6 變 0/6、poisson_time 由 4/6 變 0/6；事件 neither 為 0/4。不能把組合總分 41/48 當成全面改善。
- 本批在第一遍內就出現局部失效，限制一遍不足以保護所有技能。沒有任何全門檻過關點，因此本批不能驗證「全過立即停」的救回效果，也不能推論 3B 的能力上限。
- 所有數字是相同 112 題的選點驗證，不是獨立泛化測試。未為了改善分數追加 epoch、seed 或 teacher 呼叫。

判分複核保留原始答案與自動分：2027 第 12 步的區間式是代數等價式，給分；其餘待判定項把 Poisson 的時間乘法寫成次方，判錯。例如 `19 / 60 ** 107` 不等於 `19 * 107 / 60`。沒有替模型修補答案，亦未修改門檻或 grader fingerprint。

權重與保存：

- 兩份 adapter 各 97,307,544 bytes、392 個有限數值 tensors，SHA-256 已核對。
- 小型包 `3beethoven_first_students_download.zip`：185,311,896 bytes（約 185 MB），含兩份第 33 步 adapter、模型 revision、資料順序、原始驗證回答、各階段成績、複核與訓練 log。
- ZIP SHA-256：`9090815c7b49f82cfeba0c8b5a95eece1259f9f0eea3c526e4c4d3661f26af36`。
- 全部 12／24／33 步 adapter 及 optimizer checkpoints 另保存在 Kaggle 輸出。
- 訓練程式固定於 `3bc3e8b43b0808e6886d6c94171ab3b1008248bf`；逐項數據見 `STATS_FIRST_STUDENT_RESULTS.json`，下載資訊見 `STATS_FIRST_STUDENT_DOWNLOAD.json`。

已確認 [Kaggle 第 53 版下載頁](https://www.kaggle.com/code/trinashih/3beethoven-v0-2/output?scriptVersionId=348043163&select=3beethoven_first_students_download.zip) 顯示此檔案 185.31 MB。點擊檔案的 Download 即可；不必下載全部 3 GB 輸出。

