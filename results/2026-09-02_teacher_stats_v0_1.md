# Llama 3.3 70B Statistics Teacher Diagnostic v0.1

Date: 2026-09-02 PT

## Purpose

Evaluate whether Llama 3.3 70B is reliable enough to serve as a response-distillation teacher for a statistics / statistical-reasoning specialist student.

This benchmark was run before generating any statistics training data.

## Benchmark

- 40 multiple-choice questions
- Balanced answer positions: A=10, B=10, C=10, D=10
- Categories:
  - probability_bayes
  - distributions_expectation
  - inference_testing
  - regression_causality
  - data_reasoning

## Results

Overall accuracy: **87.5%**

| Category | Accuracy |
|---|---:|
| probability_bayes | 50.0% |
| data_reasoning | 87.5% |
| distributions_expectation | 100.0% |
| inference_testing | 100.0% |
| regression_causality | 100.0% |

Predicted-answer distribution:

- A: 10
- B: 10
- C: 7
- D: 11
- INVALID: 2

## Incorrect / invalid items

1. Two fair six-sided dice are rolled. Probability their sum is 9. Expected A, predicted D.
2. Independent events A and B with P(A)=0.6, P(B)=0.5. P(A union B). Expected C, predicted A.
3. P(A)=0.4, P(B)=0.5, P(A and B)=0.2. Expected B, output INVALID.
4. Bayes factory-machine defect problem. Expected C, output INVALID.
5. Recall from TP=90, FN=10, FP=30. Expected C, predicted B.

## Interpretation

The 70B model is not reliable enough to use as a broad statistics teacher without qualification. The main weakness is probability/Bayesian reasoning, where it scored only 50%.

However, the model was perfect on this diagnostic in three subdomains:

- distributions / expectation
- inference / hypothesis testing
- regression / causality

The next step should therefore be to test the 3B student on these same strong-teacher subdomains and look for a meaningful teacher-student capability gap before committing to a new distillation topic.

## Experimental rule

Do not generate statistics training data yet. First confirm that:

1. 70B remains strong on a harder focused benchmark in its candidate subdomains, and
2. 3B is materially weaker on the same benchmark.
