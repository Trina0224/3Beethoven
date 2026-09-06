# v0.15 paired diagnostic

Status: completed after v0.15 training; 256 responses, no new training or teacher calls.

Purpose: distinguish reading/binding, affine mean, affine variance and second-moment composition failures. Compare saved v0.14 and v0.15 adapters on the same 16 previously tested stories (8 moment, 8 affine Poisson variance). This is a post-test diagnostic, not a new held-out benchmark or a replacement for the recorded 44/64 result.

Each story has eight independent prompts: extract mean, extract variance, extract multiplier, extract offset, formulate transformed mean, formulate transformed variance, original task, and original task with generic symbolic reminders. Each response begins a new conversation; no correct subanswer is passed into a later prompt. There are 128 prompts per checkpoint, 256 responses total. The original-task condition uses the diagnostic output instruction, so it is not a byte-identical repeat of the earlier evaluation prompt.

References and prompts are fixed by `scripts/diagnose_stats_v0_15.py` before execution. Existing safe expression grading is used; all 128 reference expressions, wrong-zero responses and unresolved names were checked locally. For extraction a scalar is appropriate; for formulation a matching number alone remains pending semantic review. Review pending outputs before interpreting counts. A failed formatting response is not proof that the underlying fact is unknown.

The hinted condition explicitly supplies general mathematical rules and must remain separate from unaided performance. These probes can identify patterns but cannot by themselves establish the causal reason a model failed. No checkpoint selection uses these results.

## Reviewed results

All cells are out of eight stories. Automatic grades are preserved in `STATS_V0_15_DIAGNOSTIC_RESULTS.json`; full-output semantic decisions and flags are in `STATS_V0_15_DIAGNOSTIC_REVIEW.json`.

| Task | Moment v14 | Moment v15 | Poisson variance v14 | Poisson variance v15 |
|---|---:|---:|---:|---:|
| Read E[X] | 8 | 7 | 8 | 8 |
| Read/infer Var(X) | 8 | 8 | 8 | 7 |
| Identify multiplier | 0 | 0 | 0 | 4 |
| Identify offset | 8 | 8 | 8 | 8 |
| Formulate E[Y] | 0 | 1 | 5 | 4 |
| Formulate Var(Y) | 0 | 0 | 1 | 0 |
| Original task, no rule | 0 | 0 | 0 | 0 |
| Original task, general rules | 0 | **7** | 0 | **4** |

The strongest finding is the paired hinted result: v15 correctly binds and composes 11/16 formulas with generic rules, versus 0/16 for v14. The hint contains no numerical answers and is identical across checkpoints. This is evidence of improved rule-conditioned formulation on these stories; it is not independent rule recall or proof of general reasoning.

Raw v15 example: with mean 66, variance 509, scale 6 and offset 8, the hinted response is `6 ** 2 * 509 + (6 * 66 + 8) ** 2`. Without hints it emits `(6 * (66 + 8 / 2) + 8) ** 2`. In affine Poisson variance it often emits `3 * 62` instead of squaring the scale; a hint changes this case to `3**2 * 62`. Remaining hinted errors often add the constant offset or its square to a variance. Both general variance and second-moment rules were included in the hinted condition, so a focused variance-only reminder remains an untested alternative.

Extraction failures are prompt-sensitive: a request to identify the multiplier often causes an invented equation-solving exercise, despite successful use of that multiplier in the hinted task. This does not establish that the student cannot read the number. Conversely, the substep results do not support claiming that only final composition is broken: independent mean and variance formulation also fail frequently.

Review corrects prose-only scalar answers and accepts a correct substituted formula followed by consistent arithmetic. One automatic multiplier credit in v14 was revoked: the response hit the token limit mid-derivation and ended with a stray `3`, not a completed correct answer. Two v15 credits carry flags: one correctly states E[X] but conflates it with the transformed expectation; another has a correct final mean formula after incorrect earlier calculations. These are narrowly scoped binding/final-formulation credits, not endorsements of their explanations.

## Next training recommendation

Keep v15 as the starting checkpoint. Teach same-story contrasts for E[Y], Var(Y) and E[Y²], explicitly pairing scale versus scale-squared and offset affects mean versus offset does not affect variance. Include direct number-role identification using clear `Y=aX+b` wording alongside word problems. Gradually remove symbolic reminders from teacher-verified training examples, and evaluate aided and unaided conditions separately on a new frozen test. Do not reuse these diagnostic stories as an ostensibly fresh test or claim decomposition alone caused the current gains.

The original 44/64 test result remains unchanged. This diagnostic has not trained a new student.

## Saved artifacts

Kaggle Version 30 is Successful, with Quick Save configured to always save output. Diagnostic files are under `/kaggle/working/3beethoven_stats_v0_15_diagnostic/`. Version 29 remains the full training backup. The two adapter hashes were checked before inference; no weights were changed. Raw exported JSON SHA-256: `6af9cf1e1a300785da3352557630095cb0e08c2985219ece072c41417e4f8de2`.
