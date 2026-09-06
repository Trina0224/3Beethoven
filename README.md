# 3Beethoven

**A tiny local classical-music snob, distilled from a much larger Llama teacher.**

3Beethoven is an experiment in **response distillation / synthetic-data distillation**. The goal is to use a large cloud-hosted Meta Llama model as a teacher, then train a much smaller local Llama student to become a focused classical-music specialist.

The project is intentionally playful on the surface and rigorous underneath.

## Current experiment — 2026-09-06 PDT

**v0.14 student response distillation is complete.** The goal is a correctly
substituted numerical expression; an exact calculator handles arithmetic.
126 verified teacher examples / 31 validation examples, initialized from v0.13,
trained for 32 steps. Checkpoint selection used validation loss only.

| Same 64-question wording/parameter test | Vanilla 3B | v0.13 | v0.14 |
|---|---:|---:|---:|
| Frozen automatic formulation score | 0/64 | 5/64 | 30/64 |
| Supplemental semantic review | 8/64 | 8/64 | **33/64 (51.6%)** |
| Old multiple-choice retention | 86/240 | 128/240 | 123/240 |

All models received identical prompts. Semantic review credits mathematically
correct notation, complements, event sums and multiplication/division equivalence;
it does not repair wrong formulas. Raw scores and every response remain available.
The new student answers 25 more questions correctly than v0.13 (4.125 times as many).
Improvements concentrate in binomial/event probabilities and confidence intervals.
Second moments, affine Poisson variance and conditional waiting times remain weak;
old-test retention dropped by 5/240. This is a limited-domain result, not broad reasoning.

The earlier v0.13 96/96 result used familiar templates and a two-line prompt;
this test changes wording and uses one-line expressions. The two scores are not
interchangeable. v0.13 was procedural SFT; v0.14 adds filtered independent 70B responses.

The independent teacher diagnostic remains 43/48 after semantic review and failed
its gate. A new frozen diagnostic with general symbolic formula reminders passed
46/48, with unchanged thresholds. Neither diagnostic was used as student training data.

**Weights and outputs: Kaggle Version 28, Successful.** 912 responses, 392 finite
weight tensors and ZIP integrity verified. GitHub contains source, teacher targets,
raw results and review; the binary adapter is preserved on Kaggle.

- [Current status and limitations](docs/STATS_CURRENT_STATUS.md)
- [Training protocol](docs/STATS_V0_14_TRAINING_RUN.md)
- [All 912 predictions and verification](docs/STATS_V0_14_RESULTS.json)
- [Semantic review](docs/STATS_V0_14_SEMANTIC_REVIEW.json)
- [Recovery instructions](docs/KAGGLE_RECOVERY.md)
- [Historical v0.13 results](docs/STATS_V0_13_RESULTS.json)

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

**Historical checkpoint through v0.9.** Targeted v0.9 reached 64.06% on its frozen test, exceeding twice baseline, but missed the rotation-robustness goal and regressed slightly on the older test. At that checkpoint, v0.5 was retained as the broader comparison candidate. See the current experiment above for the subsequent formulation objective and v0.13/v0.14 status.

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

## Student failure diagnosis — v0.7 completed

Inference-only comparison of baseline and v0.5 on 24 exposed questions (576 responses): v0.5 maps a supplied correct value to its option in 96/96 cases, but solves only 41/96 original MC rotations. After explicit format review, no-choice direct answers are 4/24 correct; supplied-rule answers 5/24; supplied arithmetic expressions 6/24. Observed errors include variance scaling, probability arithmetic and interval endpoint rules. These are diagnostic conditions, not new held-out accuracy.

Short-calculation outputs were often truncated (baseline 16/24, v0.5 6/24), so their scores cannot establish unconstrained reasoning performance. The report preserves strict scores, all raw answers and every format-only correction. Both models reproduce all 96 original MC outputs.

- [Complete diagnosis and concrete errors](docs/STATS_DIAGNOSTIC_V0_7_RESULTS.md)
- [Raw responses and format audit](docs/STATS_DIAGNOSTIC_V0_7_RESULTS.json)
- [Fixed diagnostic protocol](docs/STATS_DIAGNOSTIC_V0_7_PROTOCOL.md)

No weights changed and no teacher calls were made. Results are preserved in Kaggle version 12 (347608100); GPU stopped. v0.5 remains the leading experimental candidate.

## Prompt and output-length diagnosis — v0.8 completed

On the same 24 exposed questions, compact supplied-arithmetic strict scores rose to baseline 14/24 and v0.5 18/24 (format review: 18/24 and 20/24). Extending previously truncated original-step outputs yielded 8/24 and 10/24 strict. Compact full-problem scores were 3/24 and 8/24: compact formatting alone was not universally better. These results qualify the earlier interpretation of broad arithmetic weakness and highlight problem-to-formula translation as a remaining bottleneck. There were no teacher calls or weight changes.

- [v0.8 report and limitations](docs/STATS_DIAGNOSTIC_V0_8_RESULTS.md)
- [All responses and explicit format review](docs/STATS_DIAGNOSTIC_V0_8_RESULTS.json)

The v0.8 archive and audited v0.9 preparation are preserved in Kaggle version 13 (347615356). v0.9 uses 180 new targeted training examples and a separately frozen 48-question test; its [protocol](docs/STATS_V0_9_PROTOCOL.md), [teacher corpus](docs/STATS_V0_9_TEACHER_DATA.json), and [data audit](docs/STATS_V0_9_DATA_AUDIT.md) are preserved before the final comparison.

## Targeted concise teaching — v0.9 completed

Same-run new-test MC: baseline 59/192 (30.73%), v0.5 91/192 (47.40%), v0.9 123/192 (64.06%). Two of three predefined goals met: ≥60% and ≥2×baseline. All-four correct improved from 0/48 to 8/48 to 17/48, still below the 24/48 goal.

No-choice strict scores: 9/48, 11/48, 33/48; independent format review: 18/48, 12/48, 33/48. Gains concentrate in squared expectation and conditional-uniform skills. Fraction arithmetic remains weak. Old-test v0.9 127/240 (52.92%) is below historical v0.5 133/240 (55.42%); those old controls were not rerun. This is targeted skill transfer, not proof of general mathematical improvement.

- [Complete report, costs and limitations](docs/STATS_V0_9_RESULTS.md)
- [960 student responses, teacher answers, logs and explicit format audit](docs/STATS_V0_9_RESULTS.json)
- [Model recovery and exact archive hashes](docs/KAGGLE_RECOVERY.md)

This run used 467 teacher calls and US$0.024613645 in reported response costs. Recorded cumulative usage is 1,306 calls and US$0.075317545, not an account balance or invoice.


## v0.10 completed — 2026-09-05 PDT

Installation was authorized and completed; training and all 1,200 student responses finished. Same-test numeric scores improved 13/48 to 20/48 and MC 67/192 to 87/192; old MC 127/240 to 130/240. Primary improvement/half-correct goals were not met. See [reviewed report](docs/STATS_V0_10_REPORT.md) and complete raw STATS_V0_10_RESULTS.json. Fraction arithmetic remains the principal bottleneck.

Kaggle Quick Save version 18 (script version 347636958) is Successful and contains final 93,104,858-byte 3beethoven_stats_v0_10.zip, SHA-256 470e4013b2f11ef52e6bd60736f73a1121e66e0bfe757093a8d3fd4e9affc677. Selected adapter SHA-256 14812770a7e612ab984e4ffad54bf514a3e00425655aa5adf732b975502f96f9. Restore version 18, not preparation-only version 15. GitHub stores code/data/results; binary backup status is in MODEL_BACKUP_STATUS.json.


## v0.11 completed — 2026-09-06 PDT

Hybrid exact-arithmetic SFT plus prior teacher-response rehearsal produced a mixed outcome. Independently reviewed canonical arithmetic:43/60 to39/60; equivalent representations:23/60 to31/60; both representations correct:22/60 to27/60; fresh statistical transfer:16/48 to16/48; old MC:130/240 to133/240. These are matched comparisons on the same new questions, not the previous v0.10 benchmark. Primary gain and statistical-transfer goals failed; retain both models without blanket promotion of v0.11. The selected first-epoch checkpoint was chosen by validation before tests; all816 responses and392 finite adapter tensors were verified. Zero new teacher API calls.

See [full reviewed report](docs/STATS_V0_11_REPORT.md) and [raw responses and independent format audit](docs/STATS_V0_11_RESULTS.json). Final binary locations are tracked in MODEL_BACKUP_STATUS.json. The arithmetic follow-up was pursued in v0.12; the current objective has since changed to correct formulation with arithmetic delegated to a tool. Historical arithmetic endpoints remain unchanged.
