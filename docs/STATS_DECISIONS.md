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
