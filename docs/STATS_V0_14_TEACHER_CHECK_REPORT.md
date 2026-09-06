# Teacher perturbation diagnostic — 2026-09-06 PDT

The teacher is useful but not consistently reliable on all eight formulation families.
No student v0.14 training was launched because the frozen teacher gate did not pass.

Before any calls, committed 24 pairs / 48 prompts and acceptance thresholds:
at least 44 verified responses, 20 complete pairs, and 75% in each perturbation group.
Each prompt received one independent response, no retries. Inputs excluded references.
The 64-question student test remains untouched. Diagnostic identities were checked
against all v0.14 splits. Some base prompts repeat between paired comparisons, so
48 responses are not 48 independent problem identities.

| Check | Automatic verified | Total |
|---|---:|---:|
| Wording | 13 | 16 |
| Parameter | 12 | 16 |
| Unit | 8 | 8 |
| Event | 8 | 8 |
| All | 41 | 48 |

Automatic pair score: 19/24. Two pending interval expressions are mathematically
correct: both compute the midpoint 295, then add a fifth of the half-width.
Their unused `width_divisor` assignments are incorrect, so these responses do not
establish fully consistent reasoning. The requested final-formulation criterion
credits the selected correct expression. Supplemental review yields **43/48 (89.6%)**
and **20/24 complete pairs**. The gate still fails; its threshold was not changed.

Five genuine errors remain:

- One Poisson affine-variance response computes a squared mean minus variance.
- Three second-moment responses omit the squared-mean contribution.
- One interval response divides and multiplies by the same factor, leaving the
  original endpoint instead of shrinking the interval.

This supports keeping the 70B teacher with verified filtering, but not claiming
unqualified reliability. The next targeted experiment should test a general
formula reminder for second moments and interval scaling on a new frozen set;
report it as rule-scaffolded teacher generation, separate from this independent run.
Do not silently repair rejected teacher outputs with gold formulas.

48 calls cost **US$0.001816105**, with no missing cost reports. No GPU was used.
All raw responses and original grades are in
[the result record](STATS_V0_14_TEACHER_PERTURBATION_RESULTS.json).

Prepared teacher-only normalized training targets: 126 training / 31 validation.
Each row retains the original response and selected attempt. Only teacher-owned
assignments are expanded; reference bindings never populate targets. The training
runner checks the teacher gate before loading models and cannot start on this result.
