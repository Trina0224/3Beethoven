# v0.4 position repair results

**Do not promote v0.4 over v0.3.** The repair slightly improves the exposed regression set, but loses accuracy on the newly frozen transfer probe. Preserve v0.3 as the preferred experimental checkpoint, not as a production-ready model.

## Main results

All scores average four cyclic option rotations per question.

| Metric | Baseline | v0.3 | v0.4 |
|---|---:|---:|---:|
| New probe correct, 96 presentations | 29 (30.21%) | 54 (56.25%) | 49 (51.04%) |
| New questions correct in all four rotations | 0/24 | 8/24 | 8/24 |
| New questions with a consistent semantic choice | 0/24 | 8/24 | 10/24 |
| New questions wrong in all four rotations | 2/24 | 4/24 | 6/24 |
| Exposed old set correct, 240 presentations | 86 (35.83%) | 129 (53.75%) | 134 (55.83%) |
| Old questions correct in all four rotations | 1/60 | 17/60 | 19/60 |
| Old questions with a consistent semantic choice | 1/60 | 18/60 | 22/60 |
| Old questions wrong in all four rotations | 4/60 | 11/60 | 12/60 |

Relative to v0.3, v0.4 loses 5.21 percentage points on the new probe: three presentations improve and eight regress. On the exposed old set it gains 2.08 points: 18 improve and 13 regress. These are paired presentations, not independent questions.

The new probe's increase from eight to ten semantically consistent questions includes two consistently wrong questions. Consistency alone is not correctness.

## New probe by task family

Each row contains four parameterized questions and four rotations each (16 presentations).

| Family | Baseline | v0.3 | v0.4 |
|---|---:|---:|---:|
| Poisson count across intervals | 5/16 | 8/16 | 3/16 |
| Expectation from mean and variance | 4/16 | 4/16 | 5/16 |
| Uniform second moment | 5/16 | 8/16 | 8/16 |
| Exactly one Type I error | 8/16 | 16/16 | 16/16 |
| At least one rejection under an alternative | 4/16 | 16/16 | 16/16 |
| Interval width after sample-size change | 3/16 | 2/16 | 1/16 |

Both trained models get all 32 presentations in the two testing families correct. Across the other four families, v0.3 scores 22/64 (34.38%), while v0.4 and the baseline both score 17/64 (26.56%). This pattern does not support a claim of broad improvement in statistical calculation.

## Remaining position dependence

| Gold answer position | v0.3 new, /24 | v0.4 new, /24 | v0.3 old, /60 | v0.4 old, /60 |
|---|---:|---:|---:|---:|
| A | 9 | 8 | 21 | 25 |
| B | 15 | 13 | 36 | 32 |
| C | 11 | 14 | 32 | 43 |
| D | 19 | 14 | 40 | 34 |

The repair shifts answer behavior toward C (35/96 new responses and 86/240 old responses); it does not eliminate positional sensitivity. Gold A remains the weakest new-probe position. A narrower position accuracy range does not compensate for lower total accuracy.

## Training and interpretation

- Reused the same 60 validated Llama-3.3-70B records, exact 48/12 question split, and pinned Llama-3.2-3B-Instruct base.
- Trained a fresh adapter, not a continuation of v0.3.
- Four rotated letter-only targets plus one unchanged original-order explanation per question: 240 training and 60 validation sequences.
- 36 optimizer steps, effective batch eight, learning rate 5e-5, seed 226, NF4 and LoRA r16/alpha32/dropout0.05. Runtime approximately 200 seconds.
- Validation loss at steps 12/24/36: 0.4695 / 0.3350 / 0.3158801198. Best checkpoint: 36. Training loss: 0.7394071288.
- The recipe changes both permutation coverage and answer-format weighting. Equal optimizer steps do not imply equal supervised token exposure. This is not a pure permutation ablation.
- The new set was frozen before any v0.4 model evaluation or training, at commit `e5cb18ce7e2617bfa0b540f96e2bfcd522a7e28b`; a subsequent source-mount fix changed no training or evaluation content.
- The 24 internally authored probes span only six parameterized families and were designed with knowledge of earlier weaknesses. They are not an external blind benchmark. The old 60-question set is explicitly exposed.
- Four cyclic rotations are not all 24 permutations. Do not treat 96 or 240 presentations as independent samples, or infer significance from these small differences.
- No extra teacher calls, no new teacher explanations, no hyperparameter search, and no test-based checkpoint selection. This outcome suggests investigating curriculum coverage and target weighting next; it does not uniquely establish a cause.

## Verification and artifacts

- [Frozen protocol](STATS_V0_4_PROTOCOL.md)
- [Full summary, verification and all 1,008 raw responses](STATS_V0_4_RESULTS.json)
- [Training and evaluation runner](../scripts/run_stats_v0_4.py)
- [New evaluation-only questions](../scripts/stats_holdout_v2.py)
- [Independent verification](../scripts/verify_stats_v0_4.py)
- [Kaggle recovery instructions](KAGGLE_RECOVERY.md)
- All 24 statistics tests passed locally and on Kaggle.
- Saved adapter reloaded successfully before final evaluation; all 392 adapter tensors are finite.
- Independent verification reconstructed gold letters from option text and checked every response, rotation identity, semantic index and aggregate correct count.
- Baseline and v0.3 each reproduce all 240 raw responses from the previous rotation run.
- ZIP CRC and all 30 manifest entries verified.
- Archive: `3beethoven_stats_v0_4.zip`, 92,691,603 bytes.
- Archive SHA-256: `00e31c2384a9303c68563e963a6d621ae57d877b3d308671279548d7edfb4262`.
- Adapter SHA-256: `8913ef7906f929c231148c88b3d726b3adc04bcd91c00f739a6a3d5a7533f9fc`.
- Training corpus SHA-256: `1a317bd424069981da0cd665559b1fd4b9d301553fc61a3d78f2368ae63965e7`.
- New benchmark SHA-256: `ff58995072c8f27322831c3f99d1ce474b2d76ff5ad8da765a68c731ae2cb547`.
- Base revision: `0cb88a4f764b7a12671c53f0838cd831a0843b95`.

The v0.4 archive is saved and verified in [Kaggle version 7](https://www.kaggle.com/code/trinashih/3beethoven-v0-2/output?scriptVersionId=347590242&select=3beethoven_stats_v0_4.zip), with status Successful.

The original v0.3 artifact remains in Kaggle version 5. Preserve v0.4 as a documented repair experiment; do not replace the preferred v0.3 checkpoint based on its small exposed-set gain. Further curriculum expansion should use newly validated Llama teacher responses and a separately frozen evaluation set.
