# Statistics v0.11: arithmetic follow-up

2026-09-06 America/Los_Angeles. Completed. Mixed outcome; the frozen overall success criteria were not met. Preserve both models; do not promote v0.11 as an across-the-board improvement.

## Same-question comparison after independent format review

| Evaluation | v0.10 | v0.11 | Change |
|---|---:|---:|---:|
| Canonical arithmetic | 43/60 (71.67%) | 39/60 (65.00%) | -4 |
| Equivalent representations of those same questions | 23/60 (38.33%) | 31/60 (51.67%) | +8 |
| Both representations correct | 22/60 | 27/60 | +5 |
| New statistical transfer | 16/48 (33.33%) | 16/48 (33.33%) | 0 |
| Old MC retention | 130/240 (54.17%) | 133/240 (55.42%) | +3 |

The two arithmetic representations are paired observations of 60 question groups, not 120 independent questions. Old MC is 60 questions with four answer-option rotations. New statistical questions are different from the previous v0.10 test; do not compare their 16/48 to the earlier 20/48 as a longitudinal regression. The same-test v0.10 control is 16/48 here.

Canonical arithmetic has six newly correct and ten newly wrong answers. Variants have twelve gains and four losses. Transfer has four gains and four losses: the unchanged total does not mean identical behavior.

## Operation breakdown

Each operation has 12 canonical and 12 paired-variant questions.

| Operation | v0.10 canonical | v0.11 canonical | v0.10 variant | v0.11 variant |
|---|---:|---:|---:|---:|
| Fraction multiplication | 11 | 8 | 6 | 7 |
| Fraction addition | 9 | 6 | 3 | 2 |
| Integer powers | 6 | 6 | 3 | 6 |
| Fraction powers | 11 | 11 | 9 | 9 |
| Reduction to lowest terms | 6 | 8 | 2 | 7 |

Reduction and some representation robustness improved, but simple multiplication/addition regressed. Numeric outputs hitting the fixed 256-token limit fell from 11 to 4 across all 168 prompts per model (canonical:3 to0; variants:7 to2; transfer:1 to2). This is a useful generation-behavior improvement, not proof of stronger general computation.

## Independent scoring audit

Strict automatic canonical scores were 41/60 and39/60. Two v0.10 answers (v11_test_reduce_000 and v11_test_reduce_005) correctly gave 8/11 and13/25 after inline Answer labels; they receive format-only credit. All invalid outputs for both models were inspected; no other credits were justified. Comma-formatted answers that were numerically wrong, unfinished repeated expressions, and incorrect decimal-over-decimal reductions remain wrong. Ordinary arithmetic accepts exact equivalent fractions; explicit reduction questions require lowest terms.

The saved UI review texts were matched to canonical raw exports after whitespace/JSON escaping normalization. Raw automatic flags and their pending-review marker are preserved in RESULTS.json; its independent_review field and this report provide the final reviewed outcomes. No test answers, decoding limits, weights, or checkpoint choices were changed after results were observed.

## Frozen goals

- Canonical arithmetic gain at least12/60: FAILED (actual -4).
- Canonical arithmetic at least30/60: PASSED (39/60; the control already exceeded this threshold).
- Statistical transfer gain at least6/48: FAILED (0).
- Old MC no worse than control minus4/240: PASSED (+3).

Thus this run does not meet the overall success definition and does not double the student's performance.

## What the diagnostic supports

On the fresh statistical questions, exact evaluation of the first numerical expression found 40/48 correct substitutions for v0.10 and39/48 for v0.11. Of v0.11's 39 correct initial expressions, 23 still ended with a wrong or missing final answer. All24 binomial initial expressions were correct in both models, but only5/24 final answers were correct in either model. Detection-probability final scores were11/24 for both.

This supports a persistent execution problem, alongside remaining substitution errors in detection questions. It does not show that the student learned nothing or prove that it memorizes all answers. The strong isolated fraction-multiplication and fraction-power controls also argue against describing it as uniformly unable to do school arithmetic. Isolated and statistical tasks differ in intermediate-number size and number of operations, so this experiment does not isolate a causal chain-length effect.

Variants alter wording or scale equivalent fractions; scaling also increases raw operand sizes. Their accuracy difference is a robustness stress test, not a pure memorization measurement. The user's concern was a high probability of memorization for familiar numbers, not a categorical claim of no learning.

## Curriculum, training and cost

Added400 exact algorithmic examples (80 each for five operations), plus516 unchanged rehearsal sequences from prior Llama response-distillation work. Validation:40 new plus64 rehearsal examples. New train/validation/test identities and final answer values are disjoint; equivalents stay in the same group. The60 canonical tests and48 fresh statistical questions also had zero final-answer collisions with the retained numerical rehearsal targets. This cannot rule out base-model pretraining exposure.

This is hybrid algorithmic SFT plus teacher-response rehearsal, not a new pure teacher-distillation comparison. No new teacher API calls; additional teacher cost US$0.

Continued the v0.10 LoRA, fresh optimizer, seed1111, two epochs/230 steps, LR3e-5, effective batch8, 4-bit NF4 on Llama3.2-3B-Instruct revision0cb88a4f764b7a12671c53f0838cd831a0843b95. Training took approximately1141seconds. Epoch1 validation loss0.1145005077 was better than epoch2 approximately0.1325, so checkpoint115 was selected before test evaluation. Training loss0.0819262 is not an accuracy measure. Maximum actual training sequence303tokens, below the768 cap; no truncation.

## Practical next decision

Keep v0.10 as the reference and retain v0.11 for the narrower robustness improvement. Do not simply add more of the same curriculum. The present targets provide the correct GCD without deriving it and often jump from repeated multiplication to a final integer. A more informative next pilot would explicitly teach Euclidean GCD and digit/carry multiplication, then test matched isolated substeps and their composition using new parameter groups. Any supplied-correct-intermediate diagnostic must be reported separately from unaided end-to-end accuracy. This run does not establish a capacity ceiling for a3B model.

## Preservation

Verified816 student responses and392 finite adapter tensors. Selected adapter SHA256:

`9994b0eb73cf824791ffbeb81dd08a301bb08801e2c38f38829af3cfd8618541`

Intermediate actual weights are in Kaggle20; selected model21; complete control22; canonical student answers23. Final snapshot, final ZIP checksum and independent-backup status are recorded in [MODEL_BACKUP_STATUS.json](https://github.com/Trina0224/3Beethoven/blob/main/docs/MODEL_BACKUP_STATUS.json). GitHub stores code, data, raw answers and audit; actual weight binaries are in saved Kaggle outputs and the separate ZIP backup.
