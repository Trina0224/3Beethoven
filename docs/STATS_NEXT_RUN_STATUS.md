# 統計蒸餾下一輪狀態

更新：2026-09-07 04:20 PDT（America/Los_Angeles）。

**現行狀態：離線修正與 readiness 已完成；尚未呼叫教師、尚未啟動 GPU。下一步是 24-story 付費教師 pilot。**

| 階段 | 狀態 | 固定結果／門檻 |
|---|---|---|
| 高階審核 | 已完成 | `STATS_CONSOLIDATION_REVIEW.md` |
| 96 情境參數組 | 已修正並凍結 | 71 train／25 validation；198／70 targets；每題可獨立作答 |
| 教師 pilot | 尚未開始 | 24 stories／65 targets；至少 59/65，且各 archetype 至少 80%；最多 48 calls／US$0.50 |
| 教師 full | 尚未開始 | pilot 通過才允許；各 category 至少 80%；教材累計最多 192 calls |
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

## Readiness 證據

- `STATS_CONSOLIDATION_READINESS.json`：`ready_for_paid_pilot`，9/9 檢查通過；teacher calls 0、GPU runs 0。
- 21 項離線測試通過，涵蓋自足 prompt、Poisson 假設、等待時間目標、split／parameter 衝突、112 題選點、pending gate、pilot/full gate、contract 恢復、288 題 holdout、教師子集與最終升級門檻。
- 固定 SHA-256：candidate stories `4bfe7995...f9d52`；selection `d752d4a9...a0a9`；holdout `c5515b35...43df`；teacher benchmark `af13bdde...50c9`。

執行順序：教師 pilot → gate → full corpus → gate → v15 同場 selection baseline → 兩起點×兩 seed → 288＋MC 最終評估 → 24 題教師對照 → ZIP／雜湊保存。只有 pilot 品質迫使修改研究規則，或四次結果完成時，才交回高階模型決策。
