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
