# 更正與最後一次實驗：原 v15 仍是保留學生

使用者已澄清：成功必須在約定的清楚統計列式範圍內整體勝過 v15，且原有各類能力不退步。**V57 repeat_control 的 16/16 只是局部改善，不算成功替代模型。** 先前「最後成功學生」與收束命名已撤回；下方原文僅保留歷史。

使用者已另行授權最後一次實驗。目前固定資料與程序已啟動：原 v15 基準、唯一一個從原 v15 出發的新學生、固定一輪 144 updates；新測試 18 類共 144 題，總分至少 +5 且各類不退步才算成功。[固定協議](STATS_FINAL_CLEAR_PROTOCOL.md)。尚未宣稱新學生成功，不增加第二輪。

---

# 最後成功範例：V57-repeat_control

文件更新：2026-09-07 22:15 PDT（America/Los_Angeles）

**最後採用的成功範例：`V57-repeat_control`。** 這是展示名稱；實際權重資料夾仍是 `repeat_control/adapter`，沒有重新命名權重，也沒有新增一個學生。

它是從原 v15 接續訓練、使用修正後 Llama 老師教材的 Llama-3.2-3B-Instruct 學生。在直接問法的雙事件機率列式套題中，原 v15 **12/16**，本學生 **16/16**。學生自行列出代入數值的算式，計算交給程式。

這是使用者收束目標後選定的既有成果展示：題組已曝光，屬回顧性範例，不是新盲測或通用統計能力認證。所有 16 題均列入；四組機率各問四種事件。機率組合未出現在該學生的事件教材中。原廣泛保留門檻未通過的紀錄維持不變。

## 不同學生的最終定位

| 名稱 | 最後定位 |
|---|---|
| **V57 `repeat_control`** | **本次最後採用的雙事件列式成功範例：16/16** |
| V57 `semantic_course` | 新語意課程比較組；事件 15/16，保留為實驗紀錄 |
| `prior_v55` | V55 的歷史比較學生；事件 13/16 |
| 原 `v15` | 訓練起點及較廣範圍比較基準；事件 12/16，不是本次最終展示學生 |
| 其他 v0.x／seed／checkpoint | 歷史實驗，不需依序繼續訓練 |

Kaggle Version 57 是保存版本號，不是新命名的「模型 v0.57」。`repeat_control` 雖名為對照組，確實是完成訓練的新學生，不是未訓練基線。

## 模型識別與取得

- [下載已保存的模型 ZIP](https://www.kaggle.com/code/trinashih/3beethoven-v0-2/output?scriptVersionId=348126445&select=3beethoven_semantic_course_students.zip)（Kaggle 權限與 Hugging Face 基礎模型存取權需沿用既有帳號）。
- ZIP：`3beethoven_semantic_course_students.zip`；選用其中 **`repeat_control/adapter`**。
- 基礎模型：`meta-llama/Llama-3.2-3B-Instruct`。
- 基礎模型 revision：`0cb88a4f764b7a12671c53f0838cd831a0843b95`。
- Adapter SHA-256：`8419430c6b7b58bd1c16c992964b1cedd76062020ce6651cb25e51df119f0eb9`。
- 這是 LoRA adapter，必須搭配上述基礎模型，不能當成獨立完整基礎權重；直接載入此最終 adapter，不要再疊加 v15 adapter。
- 實際訓練：seed 2027、126 updates、1,006 筆（750 筆修正教材＋256 筆既有教材重複練習），固定最後一步。
- 教材包含核對過的 70B 回答、助手修正的老師答案與題目措辭調整；屬 response distillation／SFT，不是 logits 蒸餾，也不是全數未修改的老師原文。

**工作已收束；沒有待執行的新訓練。** 不因歷史交接、未達成的舊門檻或待辦段落，自動呼叫老師、啟動 GPU、重跑 seed 或擴大題型。只有使用者另行提出新工作時再開始。

下列完整證據保留原始輸出。這次只整理文件，沒有新增模型訓練或推論。

---

# Limited-scope response-distillation success example

2026-09-07 22:10 PDT

## Outcome and stopping point

The user narrowed the objective to a successful example of learning to formulate expressions within a defined task, and requested an end to repeated experiments. This report closes that request using existing Version 57 outputs: **v15 12/16 → repeat_control 16/16**. No new training, inference, teacher calls, or GPU charges were needed for this review. No further run is scheduled by this report.

This is a retrospective demonstration selected after scores were visible, not a new blind test or proof of universal mastery. All 16 items from the existing event suite are included. They comprise four probability pairs crossed with four event targets, not 16 independent stories. The pairs are absent from this student's training event identities, including reversed pairs; the suite itself was previously exposed during development.

## Bounded task

Given two independent events and explicit activation probabilities, return a fully substituted arithmetic expression for both, neither, exactly one, or matching outcomes. The request uses direct, fixed terminology. Arithmetic execution is external. Arbitrary paraphrase understanding and unrelated statistics families are outside this demonstration.

| Target | v15 parent | Trained repeat_control |
|---|---:|---:|
| Both | 4/4 | 4/4 |
| Neither | 1/4 | 4/4 |
| Exactly one | 3/4 | 4/4 |
| Same outcome | 4/4 | 4/4 |
| Total | 12/16 | 16/16 |

## Concrete example

Two independent switches activate with probabilities 23/101 and 44/101. Find the probability that neither activates.

- Parent output: `Expression: (23 / 101) * (44 / 101)` — incorrectly models both activating.
- Trained output: `Expression: (1 - 23 / 101) * (1 - 44 / 101)` — correctly models both not activating.

The model emitted this expression itself; the review did not insert a formula or correct its output.

## Training and evidence

The selected artifact is the existing `repeat_control` adapter, not the assistant-authored semantic supplement arm. It continued the v15 Llama-3.2-3B-Instruct adapter for 126 updates on 1,006 rows: 750 corrected prior rows plus 256 repeated examples. Provenance includes verified 70B teacher targets, assistant-corrected teacher responses, and clarified or augmented prompts. It is teacher-response SFT with human-style editing, not pure unedited teacher supervision. The comparison is against its already-trained v15 parent, not a vanilla base model. It does not isolate which repair caused improvement.

Adapter SHA-256: `8419430c6b7b58bd1c16c992964b1cedd76062020ce6651cb25e51df119f0eb9`.

[Existing model archive on Kaggle](https://www.kaggle.com/code/trinashih/3beethoven-v0-2/output?scriptVersionId=348126445&select=3beethoven_semantic_course_students.zip). Load the `repeat_control/adapter` directory with the pinned Llama-3.2-3B-Instruct base revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`; this is a LoRA adapter, not standalone full base weights.

The archive contains original responses under `evaluation/v15/event_reviewed.json` and `evaluation/repeat_control/event_reviewed.json`, and the executed training rows under `source/docs/STATS_SEMANTIC_COURSE_DATA.json`. A separate standard-library AST/Fraction calculation verified all 32 answers against the elementary event formulas, confirmed identical paired prompts, and found no disagreement with the recorded mathematical grades. All 16 trained responses were also inspected for correct event structure, not merely numerical coincidence.

The original broad retention failures remain recorded: old-suite totals were 35/48 for v15 and 27/48 for repeat_control. This artifact is a successful limited example; it is not promoted as a general replacement. No old scores or gates have been rewritten.

## Complete paired outputs

| Item | Parent raw output | Trained raw output | Verdict |
|---|---|---|---|
| v20_validation_000_exactly_one_s0 | `Expression: (23 / 101) * (1 - 44 / 101) + (44 / 101) * (1 - 23 / 101)` | `Expression: (1 - 23 / 101) * (44 / 101) + 23 / 101 * (1 - 44 / 101)` | correct → correct |
| v20_validation_000_both_s0 | `Expression: (23 / 101) * (44 / 101)` | `Expression: (23 / 101) * (44 / 101)` | correct → correct |
| v20_validation_000_neither_s0 | `Expression: (23 / 101) * (44 / 101)` | `Expression: (1 - 23 / 101) * (1 - 44 / 101)` | incorrect → correct |
| v20_validation_000_same_s0 | `Expression: (23 / 101) * (44 / 101) + (1 - 23 / 101) * (1 - 44 / 101)` | `Expression: (23 / 101) * (44 / 101) + (1 - 23 / 101) * (1 - 44 / 101)` | correct → correct |
| v20_validation_001_exactly_one_s0 | `Expression: (12 / 101) * (65 / 101) + (65 / 101) * (12 / 101)` | `Expression: (1 - 12 / 101) * (65 / 101) + 12 / 101 * (1 - 65 / 101)` | incorrect → correct |
| v20_validation_001_both_s0 | `Expression: (12 / 101) * (65 / 101)` | `Expression: (12 / 101) * (65 / 101)` | correct → correct |
| v20_validation_001_neither_s0 | `Expression: (1 - 12 / 101) * (1 - 65 / 101)` | `Expression: (1 - 12 / 101) * (1 - 65 / 101)` | correct → correct |
| v20_validation_001_same_s0 | `Expression: (12 / 101) * (65 / 101) + (1 - 12 / 101) * (1 - 65 / 101)` | `Expression: (12 / 101) * (65 / 101) + (1 - 12 / 101) * (1 - 65 / 101)` | correct → correct |
| v20_validation_002_exactly_one_s0 | `Expression: (38 / 101) * (1 - 73 / 101) + (73 / 101) * (1 - 38 / 101)` | `Expression: (1 - 38 / 101) * (73 / 101) + 38 / 101 * (1 - 73 / 101)` | correct → correct |
| v20_validation_002_both_s0 | `Expression: (38 / 101) * (73 / 101)` | `Expression: (38 / 101) * (73 / 101)` | correct → correct |
| v20_validation_002_neither_s0 | `Expression: (38 / 101) * (73 / 101)` | `Expression: (1 - 38 / 101) * (1 - 73 / 101)` | incorrect → correct |
| v20_validation_002_same_s0 | `Expression: (38 / 101) * (73 / 101) + (1 - 38 / 101) * (1 - 73 / 101)` | `Expression: (38 / 101) * (73 / 101) + (1 - 38 / 101) * (1 - 73 / 101)` | correct → correct |
| v20_validation_003_exactly_one_s0 | `Expression: (36 / 101) * (1 - 62 / 101) + (62 / 101) * (1 - 36 / 101)` | `Expression: (1 - 36 / 101) * (62 / 101) + 36 / 101 * (1 - 62 / 101)` | correct → correct |
| v20_validation_003_both_s0 | `Expression: (36 / 101) * (62 / 101)` | `Expression: (36 / 101) * (62 / 101)` | correct → correct |
| v20_validation_003_neither_s0 | `Expression: (36 / 101) * (62 / 101)` | `Expression: (1 - 36 / 101) * (1 - 62 / 101)` | incorrect → correct |
| v20_validation_003_same_s0 | `Expression: (36 / 101) * (62 / 101) + (1 - 36 / 101) * (1 - 62 / 101)` | `Expression: (36 / 101) * (62 / 101) + (1 - 36 / 101) * (1 - 62 / 101)` | correct → correct |
