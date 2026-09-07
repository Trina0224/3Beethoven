# 統計蒸餾下一輪狀態

更新：2026-09-07 04:28 PDT（America/Los_Angeles）。

**現行狀態：資料與判分的離線修正完成；舊 pilot 尚待完整原答重判，付費呼叫與 GPU 執行鎖定。先前「開跑包完成／ready」的結論撤回；下方 pilot 分數是歷史自動接受數，不是教師正確率。**

## 本次修正與驗收（未增加教師呼叫）

- 原 288 題 holdout 有 36 道非法機率題；根因是用 `101..229` 的共用參數作百分比。現在按題型取樣合法機率，重建 288 題及其中 24 題教師對照；版本及雜湊已更新。舊檔仍可從 commit `d425717` 取回，不修改歷史 v15–v20 成績。
- 新增獨立 Fraction oracle：由分布／有效參數計算期望值，而非把參考式送進同一個計算器自我認證。檢查機率及文字百分比、二項整數條件、變異數非負、Poisson rate／時間、uniform 條件可成立、區間端點與參考值。
- 去重以每個子題的有效參數為準，包括等價時間單位；忽略與所求量無關的 offset。兩個非 pilot 情境重抽；24 個 pilot request 完全不變，可重用原始回答作離線診斷。
- 另抓到選點題 `v17_validation_poisson_scaled_001` 與 replay 的 `3**2*67` 重複；僅在本輪矩陣替換為 `v15_validation_poisson_scaled_000`。112 題規模、類別配額及整數門檻不變，未使用 test 題。
- grader v2 承認經檢核的 `comb(n,1)=n`、一次方、uniform「已等時間＋剩餘時間均值」與 interval 等價式；仍拒絕錯誤單位、剩餘等待取代總等待、未代入變數及只給最後數字。歷史 grader 檔案未改。
- 新 readiness 實際驗證含 replay 的 860 筆題目，合法性錯誤與跨分割有效子題重疊皆為 0；42 項離線測試通過。重跑 readiness 保留既有 38 calls／費用／42/65，不再歸零或變成付費開跑就緒。
- 新舊 grader／資料 gate 不可混用，待複核項不得放行；本次 `execution_authorization=offline_repairs_only`。原 pilot 的 23 個拒收目標尚未在完整來源紀錄上重判，不能宣稱修正後分數或 pilot 已通過。

下一步只需用已保存的 Version 49 `records/` 作離線重判，不必重新問老師：

```bash
python scripts/regrade_stats_consolidation_pilot.py \
  --source-root /path/to/extracted/3beethoven_stats_consolidation_teacher \
  --output /path/outside/source/pilot_regrade_v2.json
```

此命令不寫來源檔、不呼叫網路、不產生可放行的訓練 gate；有變更的 request 會標示不可重用。原 42/65 的歷史證據在 Kaggle Version 49 與下方表格。

限制：修正的是已確認的出題／分割／判分錯誤，不等於已證明自然語言語義全部無誤或跨句式泛化。教材仍是參數化模板；既有順序是故事內排列、故事間混合 replay，不能稱為完整的全體 1→2→3 課程。本次沒有追加教材、改 teacher prompt、改 replay 比例或更動兩起點×兩 seed 研究設計。

## 歷史 pilot 與先前修正紀錄（由上方現行狀態取代）

| 階段 | 狀態 | 固定結果／門檻 |
|---|---|---|
| 高階審核 | 已完成 | `STATS_CONSOLIDATION_REVIEW.md` |
| 96 情境參數組 | 已修正並凍結 | 71 train／25 validation；198／70 targets；每題可獨立作答 |
| 教師 pilot | 已完成、未通過 | 42/65；門檻 59/65；38 calls；reported cost 約 US$0.00402；原始證據在 Kaggle Version 49 |
| 教師 full | 已封鎖 | pilot gate 未過，程式拒絕 full |
| 四次訓練 | 尚未開始 | fresh base 與原 v15，各 seeds 2027／31415；一次曝光、最多 49 updates |
| checkpoint 選點 | 已凍結 | 12、24、最後一步；48 舊＋48 組合＋16 事件；pending 不放行 |
| 最終 holdout | 已生成並凍結 | 96 舊＋96 組合＋96 事件＝288；另有永久 MC 240 responses |
| 教師對照 | 已固定、尚未送出 | holdout 中 24 題；教材封存後才呼叫；不得回填訓練 |
| 最終保存 | 程式已備妥 | 每 run 驗證 adapter tensors、manifest、SHA-256、ZIP |

## 本次修正

- 44 道依賴前文的子題改成自足 prompt；教師單題重試也有完整條件。
- 12 組到達題補上 homogeneous Poisson process 假設。
- `v18_conditional_wait` 維持條件下的**總等待時間**，不再偷換成剩餘等待時間。
- 訓練與驗證使用不同表述家族；分母 100 的事件在文字中實際使用百分比。
- selection 改為同場原 v15 相對保留門檻；事件改用四個不同故事。
- 新 grader 隔離歷史版本，分開記錄數學正確、可執行、嚴格格式；任何 pending 先停。
- immutable contract 在開跑前即包含 parent、資料、grader、selection 與 token hash；採 `SequentialSampler`，只跑一個 epoch，不循環補足步數。
- 教師採持久 ledger；每次請求先預留 US$0.01，pilot 未過不能進 full。

## Pilot 實際結果

| archetype | 接受／規劃 | gate |
|---|---:|---|
| poisson_process | 1/6 | 未過 |
| affine_poisson | 4/9 | 未過 |
| general_moment | 9/9 | 通過 |
| uniform_wait | 0/4 | 未過 |
| binomial | 1/2 | 未過 |
| detection_events | 15/15 | 通過 |
| interval | 0/2 | 未過 |
| chain_poisson_variance | 6/6 | 通過 |
| chain_scaled_variance | 0/6 | 未過 |
| chain_second_moment | 6/6 | 通過 |

失敗不是單一原因。Poisson rate 題有真錯（把每分鐘 rate 也除以 60）；affine／scaled variance 常回傳含 `Var(X)`、`E[X]` 或等號鏈的非全數值式。另一方面，uniform midpoint、`comb(n,1)=n` 的 binomial、interval 等至少包含 grader 尚未自動承認的合法等價式。因此不能直接判定教師只有 42/65，也不能直接放寬 gate：下一步應先以保存的 23 個拒收 target 做盲式語義分類，分開「數學真錯」「格式違約」「合法等價式漏判」，再決定修 grader、修 prompt，或兩者都修後重跑一個新 pilot contract。

Kaggle 證據：[Version 49](https://www.kaggle.com/code/trinashih/3beethoven-v0-2/output?scriptVersionId=347939910&select=3beethoven_stats_consolidation_teacher.zip)。該 ZIP 含 38 次原始回答、持久 ledger、逐故事 accepted、pilot gate、usage、manifest 與 SHA-256；沒有教師 benchmark 或 GPU 結果。

## Readiness 證據

- 付費前 `STATS_CONSOLIDATION_READINESS.json` 為 `ready_for_paid_pilot`，9/9 離線檢查通過；pilot 後更新為 `pilot_failed_research_decision_required`。
- 21 項離線測試通過，涵蓋自足 prompt、Poisson 假設、等待時間目標、split／parameter 衝突、112 題選點、pending gate、pilot/full gate、contract 恢復、288 題 holdout、教師子集與最終升級門檻。
- 固定 SHA-256：candidate stories `4bfe7995...f9d52`；selection `d752d4a9...a0a9`；holdout `c5515b35...43df`；teacher benchmark `af13bdde...50c9`。

目前停止點符合協議：不進 full、不啟動 GPU。高階模型只需決定拒收分類後的修復邊界；其餘兩起點×兩 seed、112 題選點與 288＋240 最終評估規則維持不變。
