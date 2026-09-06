# Statistics pilot decision log

## After v0.7 diagnostics

User authorized recording the findings and proceeding autonomously through additional steps, including periods when they cannot use their phone. Preserve outputs and usage; do not require routine confirmations.

Established observations: v0.5 supplied-value option mapping 96/96; original MC 41/96; independent numerical answering remains poor. Observable errors include variance scaling, event arithmetic and confidence-interval endpoint rules. Some numerical answers were correct but rejected only for format. Short-calculation tests were censored by a 192-token limit (baseline 16/24, v0.5 6/24).

Decision:
1. Resolve prompting and truncation with a fixed compact-solution diagnostic and a larger-budget replication of truncated answers. No teacher calls or training in that step.
2. Freeze a separate internal evaluation before generating revised teaching data.
3. Build concise Llama-only solution targets with exact arithmetic/reference validation and independent content audit. Prefer short explicit calculations to abstract lesson cards or appended long explanations.
4. Train a fresh-base student using a recorded schedule, compare with baseline and v0.5, and preserve all results even if negative.
5. Keep old diagnostic questions classified as exposed development material. A new test can assess new instances of trained skills, but must not be called unseen-skill generalization.
6. Save model archives in Kaggle and code/data/reports in GitHub; stop GPU after preservation.

v0.5 remains the leading experimental candidate until new evidence warrants a change. Do not interpret supplied formulas or answers as independently solved test performance.

## v0.8 clarification and v0.9 preparation

Compact supplied-arithmetic strict accuracy is baseline 14/24 and v0.5 18/24 (format review 18/24 and 20/24). Longer original-step outputs yield 8/24 and 10/24 strict versus 2/24 and 7/24 under the earlier cap. Compact full problems yield 3/24 and 8/24; compact format is not universally better. The earlier broad interpretation of weak arithmetic must therefore be qualified: prompt/length conditions strongly affect results, while problem-to-formula translation remains weak. No teacher calls or weight updates were used in v0.8.

Proceed with a fresh-base v0.9 targeted curriculum: 180 train, 24 validation, 48 parameter-disjoint test questions frozen before generation. All 204 calculations and seven shared Llama rules audited; original and reference-conditioned repairs retained. Preparation is saved in Kaggle version 13 (347615356). Training and same-run baseline/v0.5 comparisons began after 41 tests passed. Keep the predefined thresholds and report negative outcomes as faithfully as positive outcomes.

## v0.9 outcome

Completed 135 steps and all 960 student responses. New MC 123/192 versus same-run baseline 59/192 and v0.5 91/192; ≥60% and double-baseline goals passed, all-four robustness 17/48 missed 24/48. Format-reviewed numeric answers are 33/48 versus 18/48 and 12/48. Old v0.9 127/240 is below historical v0.5 133/240. Preserve v0.9 as the targeted-skill candidate and v0.5 as the broader control. The evidence supports better formula selection, with fraction arithmetic and two interval-parameter errors remaining.

All 144 numeric answers independently read, 10 format-only credits explicitly recorded without changing strict scores. No more tuning on this now-exposed test. Next research should separate fraction arithmetic from option mapping and retain broader curriculum coverage, with another separately frozen evaluation if training resumes. Save final artifacts and stop GPU after confirmed preservation.

Final preservation confirmed: Kaggle version 14 (347620387) Successful, v0.9 and v0.8 ZIPs visible in that version. GPU shutdown verified by `Draft Session off (run a cell to start)`. Code, corpus, all raw results, audit, report and recovery instructions committed to GitHub. This authorized checkpoint is complete.


## v0.10 audited preparation checkpoint

Kaggle version 15 (347625570) is Successful and its output visibly contains `3beethoven_stats_v0_10.zip` (551,081 bytes; SHA-256 `f5772cb391f1bbd342a0e3f278e39a1a2bbe554ce72b215af139d9dbb0d2fb6c`). This is preparation only: no v0.10 student training or test has run. The 112 audited teacher records yield 516 training and 64 validation sequences with rehearsal; actual maximum length is 303 tokens, below the 768-token cap. Corpus and audit are committed separately. Teacher usage: 221 calls, $0.02018775, complete cost reporting.

Missing `bitsandbytes==0.50.2` blocks the frozen 4-bit training configuration. Automatic approval review rejected installation and requires action-time installation confirmation. Existing Secrets worked for teacher generation; no new notebook or Secret selection is needed. Complete preparation was saved before requesting that confirmation.


## v0.10 completed — 2026-09-05 PDT

Installation was authorized and completed; training and all 1,200 student responses finished. Same-test numeric scores improved 13/48 to 20/48 and MC 67/192 to 87/192; old MC 127/240 to 130/240. Primary improvement/half-correct goals were not met. See [reviewed report](STATS_V0_10_REPORT.md) and complete raw STATS_V0_10_RESULTS.json. Fraction arithmetic remains the principal bottleneck.

Kaggle Quick Save version 18 (script version 347636958) is Successful and contains final 93,104,858-byte 3beethoven_stats_v0_10.zip, SHA-256 470e4013b2f11ef52e6bd60736f73a1121e66e0bfe757093a8d3fd4e9affc677. Selected adapter SHA-256 14812770a7e612ab984e4ffad54bf514a3e00425655aa5adf732b975502f96f9. Restore version 18, not preparation-only version 15. GitHub stores code/data/results; binary backup status is in MODEL_BACKUP_STATUS.json.


## v0.11 started — 2026-09-05 PDT

User raised a high probability of answer memorization for familiar decimals, without claiming the student learned nothing, and authorized continuation. Frozen experiment adds 400 exact-arithmetic examples to 516 unchanged rehearsal sequences. Canonical test60, equivalent variants60 (same groups), fresh statistical transfer48; v0.10 and v0.11 compared with exact scalar scoring. Earlier rehearsal final-answer collision check found zero among arithmetic test and transfer. No new teacher calls. Protocol and frozen data committed before execution; five integrity/scoring tests pass. Two epochs/230 optimizer steps; best checkpoint selected only by validation. See STATS_V0_11_PROTOCOL.md. Results pending; no success claim.


## v0.11 completed — 2026-09-06 PDT

Hybrid exact-arithmetic SFT plus prior teacher-response rehearsal produced a mixed outcome. Independently reviewed canonical arithmetic:43/60 to39/60; equivalent representations:23/60 to31/60; both representations correct:22/60 to27/60; fresh statistical transfer:16/48 to16/48; old MC:130/240 to133/240. These are matched comparisons on the same new questions, not the previous v0.10 benchmark. Primary gain and statistical-transfer goals failed; retain both models without blanket promotion of v0.11. The selected first-epoch checkpoint was chosen by validation before tests; all816 responses and392 finite adapter tensors were verified. Zero new teacher API calls.

See [full reviewed report](STATS_V0_11_REPORT.md) and [raw responses and independent format audit](STATS_V0_11_RESULTS.json). Final binary locations are tracked in MODEL_BACKUP_STATUS.json. Next useful pilot is explicitly decomposed GCD and digit/carry multiplication, not simply more examples of the same form.


## v0.12 procedural arithmetic checkpoint — 2026-09-06 PDT

The alternative-method pilot restarts from v0.10 and teaches explicit place-value multiplication, repeated multiplication for powers, Euclidean GCD, reduced fractions and full Type I/II statistical calculation chains. It uses 560 deterministic new examples plus 516 unchanged rehearsal sequences and zero teacher calls. Training completed in 270 optimizer steps; validation selected checkpoint135 with loss0.1543260962. Kaggle Quick Save version25 was confirmed Successful with output saving enabled.

The full comparison had not finished when browser control restarted and lost its authenticated Kaggle session. v0.10's new strict baselines are arithmetic7/80, statistical transfer3/48 and old MC130/240. These low free-response baselines support the diagnosis that procedural calculation and output validity are distinct from retained option-selection performance. Do not claim v0.12 improvement until all three model summaries and independent format review are recovered. See [interim recovery record](STATS_V0_12_INTERIM.md).


## v0.17 exact mixtures and conservative rehearsal — 2026-09-06 PDT

The user authorized both adapter mixture comparisons and retraining from v15.
Three exact delta mixtures (25/50/75% v16) and three checkpoints (8/16/32) were
compared on a fresh balanced validation set, alongside both parents. Training used
256 accepted historical teacher responses, lr 5e-6 and 32 steps; zero teacher calls.
Validation selected step 8 and the 25% mixture before test. Supplemental review of
all validation expressions leaves selection unchanged.

Same new 96-question unaided results: v15 64, v16 57, mixture 65, retrained 64.
New/lost pairs: mixture 5/4; retrained 1/1. Old MC v15/retrained 127/240 each.
The six-question gain goal is missed by both; mixture also exceeds the three-loss
non-target retention limit. Do not promote either as a general replacement.

v16 correctly formulates 9/12 moments and 12/12 affine Poisson variances without
reminders in this prompt/test condition, while losing other skills. Preserve that
complementarity; this experiment does not establish a successful consolidation or
rule out other curricula, merge methods or routing. No test-time model selection,
formula hints, new teacher generation, logits or RAG were introduced.
