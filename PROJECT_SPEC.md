# 3Beethoven — Current Project Specification

## Objective

Produce a focused Llama 3.2 3B Instruct student that emits correct numerical expressions for direct statistics questions. Required training-target interface: `Expression: <fully substituted numerical expression>`. Keep arithmetic unevaluated; use a bounded exact calculator downstream. General prose generation, unfamiliar paraphrases, and multiple-choice performance are outside the current objective.

## Scope

Eighteen categories cover independent events, affine moments, Poisson counts and processes, uniform means and conditional means, binomial probabilities, and confidence-interval upper endpoints after sample-size changes. Structural diversity includes signed fractions, boundary cases, different probability denominators, input ordering, and seconds/minutes conversion.

## Method and current artifact

Use verified synthetic responses and materialized historical training replay. Teacher text is not ground truth: preserve provenance and corrections. Logit-based KD remains deferred. The latest run continued v15 using 720 new examples and 720 replay examples in alternating order, one epoch, seed 2027, learning rate 1e-5, and 180 optimizer updates.

- Base: `meta-llama/Llama-3.2-3B-Instruct`
- Base revision: `0cb88a4f764b7a12671c53f0838cd831a0843b95`
- Adapter: `kozakurayuki/3Beethoven-step180`
- Adapter revision: `682d5555b0e6115135ac7b9b5d718d2abef186de`
- Adapter SHA-256: `ff89a22f097e4db457dab287042ae36520ec6b9b36f26e5ed11fe42337c04f23`

Load this adapter directly on the pinned base. Do not stack it on v15. Use the base tokenizer, not the old tokenizer configuration from the Kaggle archive.

## Evaluation and outcome

| Metric | Original 3B Instruct | Step180 adapter |
|---|---:|---:|
| Automatically correct | 45/180 (25.0%) | **170/180 (94.4%)** |
| Strict one-line format | 26/180 | **180/180** |
| Pending review | 66 | **0** |

Paired outcomes: 127 wrong-to-right, 2 right-to-wrong, 43 both correct, and 8 neither correct. Net gain: 125 questions (+69.4 percentage points). Category totals improved in 17 of 18 categories and tied in one; none declined under the automatic scoring rule. Even crediting all 66 baseline pending answers would bring that baseline to only 111/180, below 170/180. This is a sensitivity bound, not a completed semantic review of those 66 answers.
The earlier v15-based frozen selection protocol remains `no_checkpoint_passed`: it required zero paired losses and had documented execution deviations. The owner subsequently requested a separate final comparison against the unmodified 3B model. This supports a bounded project success claim; it does not retroactively pass the earlier protocol or establish zero regressions.

This is evidence for the tested statistics-expression task, not general language ability, arbitrary mathematical reasoning, or isolated teacher-transfer causality. The run combines corrected synthetic supervision and historical replay; no ablation isolates their individual effects. The final set is now exposed and must not be reused as an untouched test for future model selection.

See the [final report](docs/STATS_DIVERSE_FINAL_BLIND_RESULTS.md), [method overview](docs/STATS_METHOD_OVERVIEW.md), and [original specification snapshot](PROJECT_SPEC.zh-TW.md). V58 remains withdrawn; its narrow-template results are historical, not the current success artifact.

[Preserved Traditional Chinese version](PROJECT_SPEC.zh-TW.md)
