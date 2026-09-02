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

**Planning / study phase.**

Implementation deliberately starts after the first distillation design choices are finalized.
