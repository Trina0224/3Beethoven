# Statistics v0.3: option-rotation diagnostic

The trained student improves average accuracy, but remains sensitive to answer position. This checkpoint evaluates the existing adapter; it does not retrain it.

| Metric | Baseline | Distilled |
|---|---:|---:|
| Correct over 240 presentations | 86 (35.83%) | 129 (53.75%) |
| Questions correct in all four rotations | 1/60 | 17/60 |
| Same semantic choice in all four rotations | 1/60 | 18/60 |
| Questions wrong in all four rotations | 4/60 | 11/60 |
| Same answer letter in all four rotations | 25/60 | 0/60 |
| Original-order raw responses matching prior run | 60/60 | 60/60 |
| Invalid / token-limit responses | 0 / 0 | 0 / 0 |

Mean accuracy increases by 17.92 percentage points. Sixteen questions newly become correct in all four rotations; none lose that status. However, 43/60 distilled questions fail at least one rotation, and 42/60 change semantic choice when options move. One question is consistently wrong across all rotations. More all-four-wrong questions after training (11 versus 4) is a material regression signal despite the aggregate improvement.

## Position and topic breakdown

Each gold answer position has 60 presentations.

| Correct answer position | Baseline | Distilled |
|---|---:|---:|
| A | 54/60 (90.0%) | 21/60 (35.0%) |
| B | 23/60 (38.3%) | 36/60 (60.0%) |
| C | 3/60 (5.0%) | 32/60 (53.3%) |
| D | 6/60 (10.0%) | 40/60 (66.7%) |

The baseline chooses A in 186/240 responses (77.5%). The trained student chooses A only 26/240 times (10.8%), and performs worst when A is correct. Training reduces the baseline's extreme A preference, but does not establish position invariance.

| Topic | Baseline correct / 40 | Distilled correct / 40 |
|---|---:|---:|
| Poisson | 13 | 18 |
| Expectation | 14 | 16 |
| Uniform | 9 | 15 |
| Type I error | 15 | 30 |
| Type II error | 20 | 32 |
| Confidence intervals | 15 | 18 |

The strongest gains are in Type I/II error questions. Arithmetic/distribution topics remain weak. Distilled accuracy across shifts 0–3 is 58.33%, 45.00%, 55.00%, and 56.67%.

## Method and limits

- Sixty existing holdout questions, each with four cyclic option rotations; two students, 480 fresh responses total.
- Same saved v0.3 adapter and pinned base revision, greedy decoding, 16-token cap, one visible T4.
- The original ordering was rerun rather than copied: both models reproduce all 60 prior raw answers.
- Protocol and scoring code were committed before this diagnostic ran, at commit `7b23a790262a4ebdd3179f8aa40e985c96529d12`.
- This is a post-hoc diagnostic on an already exposed holdout, not a new blind benchmark. Four cyclic rotations do not cover all 24 permutations.
- There are 60 question clusters, not 240 independent questions per model. No independent-sample significance claim is made.
- No additional teacher calls, no new training targets, and no retraining. Earlier teacher accuracy (52/60) used only the original ordering and is not a rotation comparison.
- Results support partial task learning plus residual positional sensitivity; they do not prove broad statistical reasoning competence or identify a unique training cause.

## Reproducibility and artifacts

- [Full summary and all 480 raw responses](STATS_ROTATION_V1_RESULTS.json)
- [Frozen protocol](STATS_ROTATION_V1_PROTOCOL.md)
- [Runner](../scripts/run_stats_rotation_v1.py)
- [Scoring tests](../scripts/test_stats_rotation_v1.py)
- All 21 project statistics tests passed locally and in Kaggle.
- ZIP CRC and all 10 manifest entries verified.
- Archive: `3beethoven_stats_rotation_v1.zip`, 39,397 bytes.
- Archive SHA-256: `c5d64c56d97c343ff6b505406bdcd847b846df126253166a0a92cd26d4d21272`.
- Adapter SHA-256: `7c3dd4513bd4f9e98ae03b9788f60a5337689de20056936cf03f7dba02bed4cf`.
- Benchmark SHA-256: `9ab52132b6070eb69281e884ce256322730b1debb58407d7ed26048856bba5c4`.
- Base revision: `0cb88a4f764b7a12671c53f0838cd831a0843b95`.
- Rotation output saved and verified in [Kaggle version 6](https://www.kaggle.com/code/trinashih/3beethoven-v0-2/output?scriptVersionId=347586668&select=3beethoven_stats_rotation_v1.zip), with the 39.4 kB archive visible and version status Successful.
- Original adapter and holdout archives remain in Kaggle version 5 (`347584475`); this diagnostic restores them as a mounted notebook input.

## Next experiment implied by the evidence

A subsequent training experiment should address answer-position dependence with training-only option permutations, preserve semantic labels, and emphasize weak distribution/arithmetic concepts. Any added explanation targets must still come from validated Llama teacher output under repository policy. Freeze a separate evaluation set before training; this exposed holdout should remain a regression diagnostic. This checkpoint does not execute that experiment or consume additional teacher budget.
