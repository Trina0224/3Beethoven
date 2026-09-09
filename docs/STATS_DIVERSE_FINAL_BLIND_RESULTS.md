# Final Evaluation — Original 3B vs Step180

The owner ran the evaluation in Colab and returned the complete raw bundle on September 8, 2026 (PDT). The final set contains 180 questions, ten in each of 18 categories, reserved from training and checkpoint selection. The later owner-authorized evaluation compares the unmodified 3B Instruct base with that same base plus the already chosen step180 adapter.

## Results

| Metric | Original 3B Instruct | Step180 adapter |
|---|---:|---:|
| Automatically correct | 45/180 (25.0%) | **170/180 (94.4%)** |
| Strict one-line format | 26/180 | **180/180** |
| Pending review | 66 | **0** |

Paired outcomes: 127 wrong-to-right, 2 right-to-wrong, 43 both correct, and 8 neither correct. Net gain: 125 questions (+69.4 percentage points). Category totals improved in 17 of 18 categories and tied in one; none declined under the automatic scoring rule. Even crediting all 66 baseline pending answers would bring that baseline to only 111/180, below 170/180. This is a sensitivity bound, not a completed semantic review of those 66 answers.

The mathematical primary score and strict formatting are separate metrics. Formatting alone is not the mathematical pass criterion.

## Category scores

| Category (10 questions each) | Original 3B | Step180 |
|---|---:|---:|
| at_least_one | 4 | 10 |
| binomial | 3 | 9 |
| both | 10 | 10 |
| exactly_one | 0 | 10 |
| interval | 0 | 9 |
| moment_mean | 4 | 10 |
| moment_second | 0 | 8 |
| moment_variance | 0 | 9 |
| neither | 7 | 9 |
| poisson_scaled | 0 | 10 |
| poisson_second | 1 | 10 |
| poisson_variance | 5 | 10 |
| process_scaled | 1 | 7 |
| process_second | 0 | 10 |
| process_variance | 1 | 10 |
| same | 0 | 10 |
| uniform_conditional | 0 | 9 |
| uniform_mean | 9 | 10 |

## Review of the ten student errors

IDs below omit the common `diverse_final_blind_` prefix.

| Question ID | Actual error |
|---|---|
| interval_001 | Adds endpoints instead of subtracting them when computing interval half-width. |
| process_scaled_003 | Omits the arrival rate. |
| process_scaled_004 | Omits the observation duration. |
| process_scaled_005 | Omits the arrival rate. |
| moment_second_004 | Incorrectly forms the squared mean term. |
| moment_variance_006 | Adds the squared mean to the variance. |
| moment_second_006 | Uses an incorrect second-moment expansion. |
| neither_007 | Omits the complement of the first event. |
| uniform_conditional_008 | Uses the original lower endpoint instead of the conditional cutoff. |
| binomial_009 | Multiplies by an extra failure-probability factor when r=n. |

These are real numerical formulation errors, not grader false negatives. `neither_007` and `binomial_009` are the two questions the baseline answered correctly and the student missed. No scores were changed during this review.

## Reproduction and evidence

- Base: `meta-llama/Llama-3.2-3B-Instruct`
- Base revision: `0cb88a4f764b7a12671c53f0838cd831a0843b95`
- Adapter: `kozakurayuki/3Beethoven-step180`
- Adapter revision: `682d5555b0e6115135ac7b9b5d718d2abef186de`
- Adapter SHA-256: `ff89a22f097e4db457dab287042ae36520ec6b9b36f26e5ed11fe42337c04f23`

Load this adapter directly on the pinned base. Do not stack it on v15. Use the base tokenizer, not the old tokenizer configuration from the Kaggle archive.

- Final split SHA-256: `9b216083cafab17f76b0c28f3e0941e9727234ede0d2d270056b64c5c2d6524a`
- Grader fingerprint: `f73c1d38cbd5e1ab9454275300907aa6c6a43c3833a303441ffb1311c18ad68b`
- [Raw result ZIP](../final_blind_eval_results.zip): 83,251 bytes; SHA-256 `4f3e57aeb129a6d43870496841fdf1b806e688ac9e8bfced12b0c5b968cbad54`
- [Machine-readable summary](STATS_DIVERSE_FINAL_BLIND_RESULTS.json)
- [Colab reproduction](STATS_DIVERSE_FINAL_BLIND_COLAB_HANDOFF.md)
- [Weights on Hugging Face](https://huggingface.co/kozakurayuki/3Beethoven-step180)
- [Historical Kaggle ZIP backup](https://www.kaggle.com/code/trinashih/3beethoven-v0-2/output?scriptVersionId=348384546): `3Beethoven_latest_step180_weights.zip`; SHA-256 `4a97c1070fd81ff2fb570699eb630043562b9b4968b799dfed6dfb410fdf036e`.

Both arms use the pinned base tokenizer and identical prompts, greedy decoding, a 160-token generation limit, and 4-bit NF4 loading. This reports a single evaluated artifact, not training-seed robustness.

## Limits of the success claim

The earlier v15-based frozen selection protocol remains `no_checkpoint_passed`: it required zero paired losses and had documented execution deviations. The owner subsequently requested a separate final comparison against the unmodified 3B model. This supports a bounded project success claim; it does not retroactively pass the earlier protocol or establish zero regressions.

This is evidence for the tested statistics-expression task, not general language ability, arbitrary mathematical reasoning, or isolated teacher-transfer causality. The run combines corrected synthetic supervision and historical replay; no ablation isolates their individual effects. The final set is now exposed and must not be reused as an untouched test for future model selection.

[Preserved Traditional Chinese version](STATS_DIVERSE_FINAL_BLIND_RESULTS.zh-TW.md)
