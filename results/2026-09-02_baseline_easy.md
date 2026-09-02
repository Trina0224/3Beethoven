# Easy baseline diagnostic — 2026-09-02

Model: `meta-llama/Llama-3.2-3B-Instruct`

Benchmark: `benchmarks/baseline_easy_v0_1.py`

## Result

- Overall accuracy: **96.7% (29/30)**
- Composer / period: 100%
- Form / structure: 100%
- History: 100%
- Style: 100%
- Terminology: 100%
- Instrumentation: 80%

Only missed item:

- Question: Which instrument is NOT normally part of the standard woodwind family?
- Correct answer: French horn
- Student prediction: Oboe

## Interpretation

The first diagnostic was too easy to provide useful headroom for a distillation experiment. Basic classical-music factual knowledge is already close to saturated in the vanilla 3B student.

Decision: freeze this benchmark as evidence, do not train on it, and move to a harder Level 2 diagnostic emphasizing harmony/counterpoint, formal analysis, orchestration/transposition, style discrimination, and historical context.

This pre-distillation decision is important: benchmark difficulty was increased before teacher-data generation, rather than choosing a favorable test after training.
