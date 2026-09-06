# v0.18: concept-depth curriculum with retained simpler problems

User requested 1 → 2 → 3 conceptual transformations, retaining simpler training rows and saving each stage. This experiment starts from exact v15, not the unsuccessful v17 mixture. It is a **procedural supervised curriculum ablation**, not a claim of new teacher-response distillation. No cloud teacher calls or costs are incurred.

## Explicit logical chains

| Track | One transformation | Two transformations | Three transformations |
|---|---|---|---|
| Poisson count variance | Poisson mean → variance | Rate × minutes → mean → variance | Seconds → minutes → mean → variance |
| Scaled variance | Given variance → affine variance | Poisson mean → variance → affine variance | Rate × minutes → mean → variance → affine variance |
| Second moment | Given mean and variance → second moment | Poisson mean → variance → second moment | Rate × minutes → mean → variance → second moment |
| Conditional total wait | Given uniform bounds → mean | Condition support → new bounds → mean | Seconds → minutes → conditioned bounds → mean |

Counts denote applications of the listed domain rules, not arithmetic operator counts, token length, or measured internal neural reasoning steps. Second moments here are unscaled; prior affine second moments remain a separate transfer target. All three variants of a parameterized story stay in the same split. The same primitive identities within each track cannot cross splits. Overlapping full-task identities from v13–17 are blocked.

Freeze 288 train, 48 validation and 96 test questions. Each depth has four equally sized tracks. Training uses 24 stories per track; validation four; test eight. Prompt exposes no formula or chain annotations. Targets retain arithmetic operations. References are checked by exact rational arithmetic. Exact numeric agreement alone does not establish formulation correctness except where the one-rule reference itself is a supplied scalar.

## Stages and convergence

1. Only depth 1: 96 rows.
2. Depth 2: 96 new rows + 48 depth-1 replay rows.
3. Depth 3: 96 new rows + 48 depth-2 + 48 depth-1 replay rows.

Each epoch generates answers to the same 48 validation questions. Score all three depths to show ability before and after introduction. Transition after at least three epochs when either (a) mastery persists for two checks, or (b) validation stops improving for three epochs. Mastery requires current-depth ≥14/16 with each track ≥3/4, and every prior depth ≥12/16. Plateau uses current-depth correct, then lower-depth correct, then weakest current track. **A plateau below mastery is reported as such; convergence is not the same as learning success.** Eight epochs is a safety budget. If it is reached without either transition condition, preserve the checkpoint and stop progression, explicitly reporting non-convergence.

Start each next stage from the last converged epoch. Save full optimizer/RNG checkpoints every epoch and a separate boundary adapter. No rollback to a test-selected model. LR 2e-5, constant schedule, effective batch 8, seed 1818, same pinned base as v17. Reset optimizer at stage boundaries. Prior broad multi-step examples are not mixed into the one-step-only first stage.

## Shuffled comparison and limits

After the curriculum's realized epoch counts are fixed, globally shuffle its exact training multiset. Train another v15 copy with the same examples, repetition counts, learning rate, optimizer updates and reset boundaries. Verify source-ID counters and step counts. Evaluate each control boundary on validation. Final test evaluates v15, every curriculum boundary and the final control. Test cannot affect stopping, replay or selection.

This is one seed and a curriculum-determined adaptive budget, not definitive statistical evidence of a universal order effect. It changes the dataset and LR relative to v17, so a v17 comparison alone cannot identify an order effect. Fresh tests measure parameter generalization within these known logical chains. They do not establish broad retention across the previous eight families, nor successful general-model replacement. Preserve pending semantic reviews and raw outputs. Do not promote based only on this curriculum test.
