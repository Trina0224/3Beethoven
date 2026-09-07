# 統計蒸餾下一輪狀態

更新：2026-09-07 02:42 PDT（America/Los_Angeles）。

**現行狀態：開跑包審核要求修正。研究規則已定案，程式／資料尚未符合；請先讀 [審核與執行決議](STATS_CONSOLIDATION_REVIEW.md)。以下舊交付清單為當時的準備成果，不代表驗收完成。**

| 項目 | 狀態 | 位置／結果 |
|---|---|---|
| 全案方向 | 已完成 | `STATS_PROJECT_REVIEW_2026_09_07.md` |
| 工作包一 | 已提交草案，審核要求修正 | `STATS_CONSOLIDATION_REVIEW.md` |
| 96 情境參數組 | 待修單題條件、假設、表述與分割 | 71 train／25 validation；268 targets |
| 教師 pilot／完整生成 | 尚未開始 | 程式在決策檔凍結前拒絕付費呼叫 |
| 兩起點×兩 seed 訓練 | 尚未開始 | 程式在協議與驗證矩陣凍結前拒絕 GPU 訓練 |
| 新獨立測試 | 僅完成藍圖 | 尚未產生具體題目 |
| 高階模型決策 | 已完成 | 規模保留；pilot 59/65；一輪最多 49 步；112 題相對基準選點；24 題教師對照 |

## 已完成檔案

- `scripts/stats_consolidation_pilot.py`：可重現故事、先分割、教師請求與選點矩陣。
- `scripts/prepare_stats_consolidation_teacher.py`：可恢復、兩次嘗試、call／cost gate、raw／拒收保存；本輪只跑過 `--scope validate-only`。
- `scripts/run_stats_consolidation_compare.py`：四個 run 共用的一次執行器；新 LoRA／原 v15 雙起點、線上首次通過即停、恢復與隔離保存。
- `scripts/test_stats_consolidation_pilot.py`、`scripts/test_stats_consolidation_grader.py`：10 項離線測試全數通過。
- `docs/STATS_CONSOLIDATION_CANDIDATE_STORIES.json`：96 故事、268 個可獨立計算的參考目標。
- `docs/STATS_CONSOLIDATION_PILOT_REQUESTS.json`：不含參考答案的教師請求草案；最大 request 1,856 bytes。
- `docs/STATS_CONSOLIDATION_COVERAGE.json`：概念與分割計數。
- `docs/STATS_CONSOLIDATION_SELECTION_VALIDATION.json`：91 題 checkpoint 選點矩陣，未凍結。
- `docs/STATS_CONSOLIDATION_HOLDOUT_BLUEPRINT.json`：獨立測試規格，沒有具體題目。
- `docs/STATS_CONSOLIDATION_DECISION_DRAFT.json`：所有執行參數與待定項目；狀態仍為 draft。

## 驗證紀錄

- `python stats_consolidation_pilot.py`：成功，固定產出 96 stories／268 questions／24-story pilot。
- `python prepare_stats_consolidation_teacher.py --scope validate-only`：成功；24 pilot stories、最大 request 1,856 bytes，零 API call。
- `python -m unittest test_stats_consolidation_pilot.py test_stats_consolidation_grader.py`：10/10 通過。
- `python -m py_compile ...`：三個新執行腳本通過。
- 以 draft 決策檔嘗試訓練入口：如預期在載入 GPU／教師資料前停止，錯誤為 `Protocol remains a draft; GPU training is blocked`。

下一個動作：中階模型依 `STATS_CONSOLIDATION_REVIEW.md` 修正並驗收，保存 readiness 證據後凍結執行設定；按現有授權執行 pilot。pilot／逐類接受門檻通過後繼續全批、四次訓練、測試與保存。一般工程修正不需再次交回高階模型；需要改研究規則、pilot 品質不過或四次結果完成才交回。
