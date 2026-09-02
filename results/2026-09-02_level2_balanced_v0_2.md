# Level 2 Balanced Baseline v0.2 — 2026-09-02

Model: `meta-llama/Llama-3.2-3B-Instruct`

This benchmark was created before teacher-data generation and is now frozen. It must not be included in any training or distillation dataset.

## Expected answer distribution

- A: 12
- B: 13
- C: 13
- D: 12

## Baseline result

Overall accuracy: **74.0% (37/50)**

| Category | Accuracy |
|---|---:|
| Harmony / Counterpoint | 40% |
| Orchestration | 70% |
| Form / Analysis | 80% |
| History / Context | 80% |
| Style Comparison | 100% |

Predicted answer distribution:

- A: 24
- B: 12
- C: 7
- D: 7

The balanced answer key exposed a substantial **A-position bias** in the vanilla 3B model. This is a useful evaluation finding and one reason the earlier v0.1 result (84%) should not be used as the primary pre/post benchmark.

## Incorrect items

1. Harmony/counterpoint — dominant seventh pitches in C major — expected C, predicted A
2. Harmony/counterpoint — cadential 6/4 — expected D, predicted A
3. Harmony/counterpoint — oblique motion — expected C, predicted A
4. Harmony/counterpoint — normally avoided voice-leading motion — expected B, predicted A
5. Harmony/counterpoint — retardation vs suspension — expected D, predicted A
6. Harmony/counterpoint — preparation of a 4-3 suspension — expected C, predicted A
7. Form/analysis — simple binary form — expected D, predicted A
8. Form/analysis — rounded binary form — expected C, predicted A
9. Orchestration — English horn in F transposition — expected C, predicted B
10. Orchestration — viola standard tuning — expected D, predicted A
11. Orchestration — contrabassoon transposition — expected C, predicted A
12. History/context — Schumann's journal — expected B, predicted A
13. History/context — Mahler's Vienna post — expected D, predicted A

## Interpretation

This is a strong baseline for the distillation experiment because it leaves meaningful headroom without being too difficult for the student model.

Primary weakness areas:

1. **Harmony / counterpoint** — highest priority for teacher curriculum
2. **Orchestration** — second priority, especially transposition and notation facts
3. **Form / analysis** — second priority, especially formal distinctions
4. **History / context** — moderate room for improvement
5. **Style comparison** — currently saturated on this benchmark; teacher-data budget should not be concentrated here

## Experimental rule

`level2_balanced_v0_2.py` is now the primary frozen held-out benchmark for before/after distillation comparison. Do not edit its questions to improve post-training scores, and do not include any benchmark question or near-duplicate in teacher-generated training data.
