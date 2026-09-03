# Statistics Teacher/Student Gap — 2026-09-02

## Purpose
Identify a domain where Llama 3.3 70B is materially stronger than Llama 3.2 3B before generating any distillation training data.

## Frozen 40-question benchmark
Five categories, 8 questions each, with balanced expected answer positions: A=10, B=10, C=10, D=10.

## Results

| Category | Llama 3.3 70B | Llama 3.2 3B | Gap |
|---|---:|---:|---:|
| probability_bayes | 50.0% | 37.5% | 12.5 pts |
| distributions_expectation | 100.0% | 62.5% | 37.5 pts |
| inference_testing | 100.0% | 50.0% | 50.0 pts |
| regression_causality | 100.0% | 100.0% | 0.0 pts |
| data_reasoning | 87.5% | 75.0% | 12.5 pts |
| **Overall** | **87.5%** | **65.0%** | **22.5 pts** |

## Student answer-position behavior
The 3B student's predictions were skewed toward A despite the benchmark having balanced ground-truth positions:

- A: 17
- B: 11
- C: 7
- D: 5

This should be monitored in later evaluation.

## Interpretation
Two categories show strong teacher competence and substantial student headroom:

1. **Inference / hypothesis testing** — 70B: 100%, 3B: 50%, gap: 50 points.
2. **Distributions / expectation** — 70B: 100%, 3B: 62.5%, gap: 37.5 points.

These are the strongest current candidates for response distillation.

Categories not recommended as primary targets:

- **Probability / Bayes:** teacher itself scored only 50%; unsuitable as a trusted source of synthetic labels.
- **Regression / causality:** both models scored 100%; no useful distillation headroom on this benchmark.
- **General data reasoning:** gap is modest and teacher is not perfect.

## Decision
Do not generate broad statistics training data. If this domain is pursued, constrain the first distillation curriculum to inference/hypothesis testing and distributions/expectation, while preserving this benchmark as held-out evaluation data.

## Benchmark integrity
This benchmark must remain excluded from all training-data prompts, generated examples, fine-tuning datasets, and curriculum templates. It is a frozen held-out evaluation set.
