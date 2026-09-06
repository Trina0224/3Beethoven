# 3Beethoven

**A tiny local classical-music snob, distilled from a much larger Llama teacher.**

3Beethoven is an experiment in **response distillation / synthetic-data distillation**. The goal is to use a large cloud-hosted Meta Llama model as a teacher, then train a much smaller local Llama student to become a focused classical-music specialist.

The project is intentionally playful on the surface and rigorous underneath.

## Core idea

```text
Large cloud Llama teacher
        ↓
Generate structured classical-music examples
        ↓
Validate and filter teacher output
        ↓
Fine-tune / response-distill a local ~3B Llama student
        ↓
Compare:
  vanilla student vs distilled student vs teacher
```

## What the student should learn

The first version focuses on mature, well-documented classical-music knowledge:

- historical periods and chronology
- composers and representative works
- musical forms and genres
- instrumentation and orchestration concepts
- terminology
- stylistic comparison using observable/historical features
- correction of common misconceptions

The project avoids subjective "who is greatest?" judgments as evaluation targets.

## Why 3Beethoven?

The joke is the mismatch between size and attitude: a roughly **3B-parameter** local model with the confidence of an unbearable conservatory expert.

It may say things like:

> You called Bach Romantic. My three billion parameters would like a word.

But the joke is presentation only. Factual accuracy is evaluated separately.

## Research question

> Can a small local Llama inherit useful classical-music expertise from a much larger cloud Llama through response distillation?

## Planned evaluation

At minimum, compare:

1. vanilla local student
2. distilled local student
3. cloud teacher

Candidate metrics include factual accuracy, classification accuracy/F1, misconception correction, explanation quality, hallucination/error rate, latency, memory footprint, and model size.

## Distillation paths

### Phase 1 — current

**Response distillation / synthetic-data distillation**

The teacher generates high-quality structured examples; the local student trains on filtered teacher responses.

### Future

**Logit-based knowledge distillation**

A later experiment may use teacher logits/soft targets, temperature scaling, softmax distributions, KL divergence, and distillation loss.

## Important project rules

See [`AGENTS.md`](AGENTS.md) before making changes.

In particular:

- no `Co-authored-by:` commit trailers
- no AI/bot authorship metadata
- no NDA or company-internal material
- no secrets or API keys
- no claims based only on cherry-picked demos
- preserve a held-out evaluation set

Full experiment design: [`PROJECT_SPEC.md`](PROJECT_SPEC.md)

## Status

**Statistics pilot through v0.6 completed. v0.6 did not improve fresh-test accuracy; v0.5 remains the leading statistics research candidate; v0.3 remains preserved for comparison and recovery.** The classical-music specialist remains the overall project goal. The current statistics pilot tests the training/evaluation pipeline; it does not establish classical-music capability.

| Checkpoint | Baseline | Trained v0.3 | Interpretation |
|---|---:|---:|---|
| Exposed development set, 24 questions | 33.3% | 75.0% | Development result |
| New holdout, original order, 60 questions | 35.0% | 58.3% | Internally authored evaluation |
| Same holdout, four cyclic rotations | 35.8% | 53.8% | Position-sensitivity diagnostic |
| All four rotations correct | 1/60 | 17/60 | Robustness remains limited |

- [Training report](docs/STATS_V0_3_RESULTS.md)
- [Holdout report](docs/STATS_HOLDOUT_V1_RESULTS.md)
- [Rotation report and raw responses](docs/STATS_ROTATION_V1_RESULTS.md)
- [Teacher policy](docs/TEACHER_POLICY.md)

Code and text results are preserved in GitHub. The adapter and holdout archives are in Kaggle version 5 (347584475); rotation results are in version 6 (347586668). Total recorded teacher usage through this checkpoint is 154 calls and $0.007196965 in response-reported costs, not a billing statement. Rotation evaluation added no teacher calls.

The v0.4 repair reused the same teacher records with rotated letter targets. On a new 24-question, four-rotation probe, baseline / v0.3 / v0.4 scored 30.21% / 56.25% / 51.04%. The old-set gain (53.75% to 55.83%) did not carry over to this new probe; v0.4 is not promoted. Both trained models answered all rotations correctly for 8/24 new questions.

- [v0.4 complete results and all 1,008 responses](docs/STATS_V0_4_RESULTS.md)
- [v0.4 frozen protocol](docs/STATS_V0_4_PROTOCOL.md)
- [Kaggle recovery and resume instructions](docs/KAGGLE_RECOVERY.md)

The v0.4 model archive is preserved in Kaggle version 7 (347590242). This repair added zero teacher calls. The evidence supports partial transfer with remaining position sensitivity and weak arithmetic generalization, not a claim that distillation failed or that the teacher is inadequate. A subsequent study should examine training coverage and target weighting with a separately frozen evaluation.

## Expanded curriculum experiment — completed

v0.5 expanded to 180 training questions across 18 task families, 24 validation questions and 36 frozen test questions. All 204 teacher records were independently read before training; 27 records were revised by Llama to correct false or misleading explanations. Final student explanation targets remain Llama-only.

| Same-run comparison | Baseline | v0.3 | v0.5 |
|---|---:|---:|---:|
| New 36 questions × four rotations | 40/144 (27.78%) | 52/144 (36.11%) | 72/144 (50.00%) |
| New questions all four correct | 0/36 | 0/36 | 2/36 |
| Old 60 questions × four rotations | 86/240 (35.83%) | 128/240 (53.33%) | 133/240 (55.42%) |
| Old questions all four correct | 1/60 | 17/60 | 19/60 |

All six new topic totals improved, but position sensitivity remains substantial. These are six parameterized test families, not 144 independent questions. The run increased curriculum coverage and training compute; it is not a compute-matched ablation. Training ran 135 steps, while validation selected the first-epoch checkpoint at step 45. Later validation losses worsened.

The current v0.3 old-set result differs from historical 129/240 by one answer; the full report preserves and identifies the difference. Historical results above are not overwritten.

- [Complete v0.5 report](docs/STATS_V0_5_RESULTS.md)
- [All 1,152 model responses, runtime versions and logs](docs/STATS_V0_5_RESULTS.json)
- [Teacher data audit and costs](docs/STATS_V0_5_DATA_AUDIT.md)
- [Audited teacher records and provenance](docs/STATS_V0_5_TEACHER_DATA.json)
- [Frozen v0.5 protocol](docs/STATS_V0_5_PROTOCOL.md)
- [All frozen questions](docs/STATS_V0_5_FROZEN_QUESTIONS.json)
- [Kaggle recovery instructions](docs/KAGGLE_RECOVERY.md)

Kaggle version 8 (347596444) preserves the pre-training corpus; version 9 (347598932) preserves the complete model and verified ZIP. This run used 572 teacher calls and $0.037577505 in response-reported costs. Across recorded experiments: 726 calls and $0.044774470, not an account balance or invoice. Model archives remain on Kaggle; code and text results are in GitHub.

## Same questions, paired teaching — v0.6 completed

v0.6 reused the audited v0.5 questions and paired related Llama explanations. Independently rejected abstract lesson cards were preserved but excluded from training. The frozen new test and success thresholds were unchanged.

| Same-run comparison on the v0.6 test | Baseline | v0.5 | v0.6 |
|---|---:|---:|---:|
| New 48 questions × four rotations | 55/192 (28.65%) | 79/192 (41.15%) | 79/192 (41.15%) |
| New questions all four correct | 0/48 | 10/48 | 9/48 |
| Old 60 questions × four rotations | 86/240 (35.83%) | 133/240 (55.42%) | 140/240 (58.33%) |

**No fresh-test gain; all three success thresholds were missed. v0.6 is not promoted.** These 48 questions differ from the v0.5 test above. The teacher scored 30/48 (62.50%) in original order; that is not a rotation average. Sequence count and optimizer schedule matched v0.5, but longer paired targets added tokens and repeated exposure, so this is not a compute-matched ablation.

- [Complete v0.6 report](docs/STATS_V0_6_RESULTS.md)
- [All 1,296 student responses, teacher answers and logs](docs/STATS_V0_6_RESULTS.json)
- [Data audit and protocol deviation](docs/STATS_V0_6_DATA_AUDIT.md)
- [Frozen protocol](docs/STATS_V0_6_PROTOCOL.md)
- [Recovery instructions](docs/KAGGLE_RECOVERY.md)

Kaggle version 10 (347602173) preserves preparation; version 11 (347605195) preserves the selected adapter and verified ZIP. This run added 113 teacher calls and $0.005929430 in response-reported costs. Cumulative recorded usage is 839 calls and $0.050703900, not an account balance or invoice. The GPU session was stopped after successful preservation.
