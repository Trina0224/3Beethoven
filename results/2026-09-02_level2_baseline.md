# Level 2 Baseline Diagnostic — 2026-09-02

Model: `meta-llama/Llama-3.2-3B-Instruct` (vanilla, pre-distillation)

## Result

- Overall: **42/50 = 84.0%**
- Harmony / counterpoint: **6/10 = 60%**
- Form / analysis: **8/10 = 80%**
- Orchestration: **8/10 = 80%**
- Style comparison: **10/10 = 100%**
- History / context: **10/10 = 100%**

## Missed items

1. 4–3 suspension resolution
2. Conventional deceptive cadence in major
3. Contrary motion in species counterpoint
4. Italian augmented-sixth chord in C major
5. Typical secondary-key area in a Classical major-key sonata exposition
6. Baroque da capo aria form
7. Horn in F transposition
8. `sul ponticello`

## Interpretation

The vanilla 3B model is already saturated on broad style/history recall, while the clearest headroom appears in harmony/counterpoint, followed by formal analysis and orchestration/notation.

However, **v0.1 has an answer-position imbalance**: every style-comparison item and every history/context item has `A` as the correct answer. Several wrong answers elsewhere were also predicted as `A`. Therefore the 84% score is useful diagnostically but should **not** be treated as the final frozen benchmark for before/after claims.

Next action: create a balanced Level 2 v0.2 (same skill domains, new held-out questions, approximately uniform A/B/C/D answer positions) before generating any teacher training data.

## Experimental integrity

- This benchmark was run before teacher-data generation.
- Benchmark questions must not be copied into synthetic training data.
- v0.1 remains preserved as historical evidence; it should not be silently edited after seeing results.
