# 統計蒸餾下一輪狀態

更新：2026-09-07 05:45 PDT（America/Los_Angeles）。

**現行狀態：教師 pilot 已完成但 gate 失敗；full 與 GPU 均未啟動。這是交回高階模型的研究決策點。**

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
