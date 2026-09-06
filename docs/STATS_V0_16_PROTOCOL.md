# v0.16: same-story contrasts and fading formula support

Frozen before teacher generation and student evaluation. Source student: v0.15 adapter SHA-256 `9369d52de4a886df9da0c872cd41bd4e01af0a38bf02ad724b5951c1a6b9f5d3` (Kaggle Version 29). Curriculum digest: `db2a2a3513e680ca64cea2d2a80e84f23e332377b140bcbc5150ccd0311c0f6c`.

## Rationale

The v15 paired diagnostic improved from 0/16 to 11/16 under symbolic rules compared with v14, while unaided performance stayed 0/16. Independent number-role and variance probes also failed. This run trains task selection and numerical binding through same-story contrasts, then removes support. No claim that only formula composition is broken.

## Corpus and separation

- 48 new training stories: 24 general affine moments, 24 affine Poisson stories. Each asks multiplier, transformed mean, transformed variance and second moment: 192 teacher targets.
- 12 separate validation stories with three formulation tasks each: 36 candidates, always validated without hints.
- 96 new primary test questions, 12 per each of the eight existing families. Full problem identities are excluded from v13–v15 and both v14 teacher diagnostics. Story/task variants never cross splits.
- 48 replay examples exclusively from verified v14 training records, six per family. No evaluation outputs enter training.
- Known mathematical families and wording templates; this is new-parameter evaluation, not an unseen-concept or unseen-template benchmark. Equal numerical answers may occur.

## Teacher

Llama 3.3 70B receives the question plus a focused symbolic rule for the requested task. It does not receive numeric gold, grader feedback containing numeric answers, or student test questions. Maximum three responses per candidate, 684 calls, with a conservative US$0.30 reported-cost guard. Cache and ledger persist. Preserve all rejected responses. Train only on verified teacher-owned expressions; no numerical repair. Require at least 180 training targets, at least 42 per task, and 30 validation targets. Grader remains the frozen v15 formulation-v2 implementation; unmatched equivalents are pending review.

## Training

Start v15; QLoRA configuration inherited. Effective batch eight, learning rate 2e-5, seed 1616. Stage 1: full task-specific symbolic reminder, one epoch. Stage 2: requested-quantity cue without formula, one epoch. Stage 3: no reminder, three epochs. Each stage uses accepted new targets plus 48 replay examples. Optimizer and cosine schedule restart per stage. Validation is always no-reminder; load lowest-validation-loss checkpoint within each stage. The final candidate is selected within the final stage, never by test score. About 150 optimizer steps with complete acceptance.

This bundles contrastive tasks, support fading and additional compute; it is not a compute-matched ablation. Scale extraction targets may be scalar; all formulation targets require the numerical expression structure.

## Evaluation and completion

Evaluate v15 and v16 with identical no-hint prompts on the fresh 96, separately with focused task-specific rules on the 24 moment/Poisson-variance questions, and on the old 240 option rotations. Total 720 saved responses. Current aided prompt is focused by task, unlike the previous diagnostic's combined rules; compare checkpoints within this run, not aided percentages across runs.

Primary goal: improve the paired unaided total and show gains in both weak families. Working target: at least 8/12 correct in each weak family, with no more than three lost correct answers across the other six families. Missed targets will be reported without redefining them. Preserve automatic and supplemental semantic scores separately; review complete raw responses before conclusions.

Verify all 720 predictions, 392 finite adapter tensors, file manifest and ZIP contents. Save weights/output on Kaggle and code, teacher records, reports and raw answers on GitHub. Stop GPU only after successful saved output. Do not stop after teacher preparation; continue through student training, evaluation, review and backup.
