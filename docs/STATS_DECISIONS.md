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
