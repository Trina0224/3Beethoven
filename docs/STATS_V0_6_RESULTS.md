# v0.6 paired-teaching experiment — no fresh-test gain

The completed experiment did **not** meet any preregistered success threshold. Keep **v0.5 as the leading experimental candidate**; retain v0.6 as a negative research result, not a promoted model.

## Same-run results

All three students answered the same frozen 48 new questions and 60 previously exposed questions, each under four cyclic option rotations.

| Metric | Baseline | v0.5 | v0.6 |
|---|---:|---:|---:|
| New questions, 192 responses | 55 (28.65%) | 79 (41.15%) | 79 (41.15%) |
| New questions, all four correct | 0/48 | 10/48 | 9/48 |
| New questions, same semantic choice across rotations | 0/48 | 13/48 | 12/48 |
| Old questions, 240 responses | 86 (35.83%) | 133 (55.42%) | 140 (58.33%) |
| Old questions, all four correct | 1/60 | 19/60 | 19/60 |
| Old questions, same semantic choice across rotations | 1/60 | 23/60 | 22/60 |

Semantic consistency can include consistently wrong answers. On new responses, two v0.5 errors became correct and two correct answers became wrong; eight raw responses changed. On old responses, ten errors became correct and three correct answers became wrong; eighteen raw responses changed.

| Frozen success criterion | Required | v0.6 | Met |
|---|---:|---:|---|
| New accuracy | >=60% | 41.15% | No |
| Double same-run baseline | >=57.29% | 41.15% (1.44× baseline) | No |
| New questions correct under every rotation | >=24/48 | 9/48 | No |

The Llama teacher answered **30/48 (62.50%)** in original order. On that same original-order slice, baseline / v0.5 / v0.6 answered 15 / 22 / 22 out of 48. Teacher results are not a four-rotation average and must not be substituted with teacher scores from earlier, different tests.

## Where results changed

Each new topic contains eight questions × four rotations.

| New topic | v0.5 correct /32 | v0.6 correct /32 |
|---|---:|---:|
| Poisson | 15 | 16 |
| Expectation | 7 | 7 |
| Uniform | 32 | 32 |
| Type I error | 6 | 6 |
| Type II error | 10 | 9 |
| Confidence intervals | 9 | 9 |

The uniform-family result is narrow: it does not establish general mathematical competence. The old-set gain does not establish improved fresh transfer.

## Actual teaching intervention and audit deviation

The initial plan generated abstract lesson cards. Independent audit found false or ambiguous mathematical claims, including an unnecessary independence condition for linearity of expectation and incorrect probability examples, even after same-teacher review. All 65 calls and raw cards were preserved; **none of those cards were used for training**.

Before student evaluation or training, the protocol was amended to pair existing, independently audited v0.5 Llama explanations. The 48-question test and success thresholds remained frozen. Within each topic and parameter index, task families were paired cyclically 0→1→2→0.

The 180 primary training questions remain unchanged. Explanation examples add a related training question to the prompt and append its original Llama explanation under a mechanical heading. Letter-only targets and all 24 validation questions remain unchanged. Mathematical assistant-target text comes from audited Llama responses; no ChatGPT mathematical explanations were inserted.

There are still 360 training sequences and 135 optimizer steps, but target characters increase from 48,941 to 76,464, and prompt characters from 85,268 to 118,816. Every training question receives additional exposure as a contrast. This is **not a token- or compute-matched ablation**: presentation, context length and repeated exposure changed together. See the [data audit](STATS_V0_6_DATA_AUDIT.md) and [amended frozen protocol](STATS_V0_6_PROTOCOL.md).

## Training and verification

A fresh base model used the same pinned revision, seed 226, learning rate 5e-5, effective batch 8, three epochs, NF4 and LoRA rank 16 / alpha 32 / dropout 0.05. Training used one visible Tesla T4 within the Kaggle T4×2 allocation.

Training completed 135 steps in 768.0669 seconds. Validation selected **step 45, epoch 1**. Validation losses were 0.7114528, 0.9615400 and 1.1395471 across the three epochs; final training loss was 0.3715316. Later fitting did not improve validation. Evaluation reloaded the selected saved adapter.

The complete export contains **1,296 student responses**, the 48 teacher answers, environment versions and trainer logs. All responses were independently rescored and option mappings checked; no invalid student answers or token-limit hits occurred. Both baseline and v0.5 reproduce all 240 old raw responses exactly. Artifact checks validated 392 finite adapter tensors, ZIP CRC and 193 manifest entries. The deterministic test suite passed 33 tests before training; the four v0.6 tests also passed after the standalone preparation recovery fix.

## Costs

This run made **113 teacher calls**, with response-reported cost **$0.005929430**: 65 discarded-card generation/review calls cost $0.005038840 and 48 teacher test calls cost $0.000890590. All 113 responses supplied cost fields.

Across recorded experiments, cumulative usage is **839 calls, $0.050703900**. These are recorded API response costs, not an invoice, credit balance or Kaggle GPU charge.

## Preserved artifacts

- [Complete results and raw responses](STATS_V0_6_RESULTS.json)
- [Frozen test](STATS_V0_6_FROZEN_TEST.json)
- [Preparation and teacher usage](STATS_V0_6_PREPARATION.json)
- [Recovery instructions](KAGGLE_RECOVERY.md)
- Kaggle version 10 (347602173): preparation, paired corpus, rejected cards and teacher test.
- Kaggle version 11 (347605195): selected adapter and complete verified archive. Quick Save was confirmed Successful and the ZIP was present in saved output.

Archive: `3beethoven_stats_v0_6.zip`, **92,937,229 bytes**.

| Artifact | SHA-256 |
|---|---|
| ZIP | `903d4cc86ea6ce97a47b5a7afbbb5abdef1667c256266d90ed816f1bb2844d5d` |
| Selected adapter | `01af62d35b4e0361e3768d6e530aceeed7d3744097d234601a63faa4dfcf02b7` |
| Frozen test | `9798cdb9ab8281f45dbb7cd14219c365a6231a6b8eebbe43a46ea858f30b51e8` |
| Paired training records | `47bfda6f0a6d7255e228dcc0c5bb6534ab3229a93b03c0ece7b4a31603d1a37a` |
| Unchanged validation records | `3689fc58f7e0a2b1b78a7e9fb2292ccb6e830e056797d76a06e886b12af890e5` |
| Base revision | `0cb88a4f764b7a12671c53f0838cd831a0843b95` |

## Interpretation limits

The 48 internally authored questions cover six parameterized families. Four rotations of a question are correlated observations, not 192 independent questions. These are new stories/parameters using known principles, not an external blind benchmark or unseen skills. The old 60 questions have been repeatedly exposed during project development. This is a single-seed experiment; no significance or broad superiority claim is warranted.

This intervention failed to improve fresh accuracy and slightly reduced rotation robustness. It does not prove that all alternative teaching methods or response distillation fail. Any subsequent training proposal should freeze a new evaluation rather than tune against this now-exposed set.
