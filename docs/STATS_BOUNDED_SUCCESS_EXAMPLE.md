# 舊局部結果：V57 與 V58 均未獲全面升級

**解讀修正：V58 的採用與成功定位也已撤回。** V57 的 v15 12/16 → repeat_control 16/16 仍是已曝光事件題上的歷史局部結果；V58 的 v15 107/144 → final_clear 144/144 仍是歷史窄模板結果。兩者都不足以證明全面優於 v15，目前沒有採用的新學生。

目前結論見 [目前狀態](STATS_CURRENT_STATUS.md) 與 [V58 撤回後記](STATS_V58_POSTMORTEM.md)。原 v15 是唯一可信的起點與比較錨點，不是成功終點；新的受控重做仍在訓練前，尚無新學生。下方保留原始實驗數字及當時敘述供追溯，並非目前模型選擇、現行 gate 或下一輪指令。

<details>
<summary>歷史紀錄</summary>

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

</details>
