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

**Statistics pilot and v0.4 repair completed; retain v0.3 as the preferred experimental checkpoint.** The classical-music specialist remains the overall project goal. The current statistics pilot tests the training/evaluation pipeline; it does not establish classical-music capability.

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
