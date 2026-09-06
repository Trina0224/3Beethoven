# 3B failure diagnosis — v0.7 (inference only)

## Finding

**Follow-up correction:** [v0.8](STATS_DIAGNOSTIC_V0_8_RESULTS.md) shows much stronger supplied-arithmetic performance with a compact calculation prompt (v0.5 18/24 strict, 20/24 after format review), and resolves the earlier length caps. The direct-answer failures below must not be interpreted as a general inability to calculate. Original observations and scores are preserved.

The v0.5 student reliably maps a **supplied correct value to an option**, but often fails to derive that value independently. Its observed failures include selecting/applying mathematical rules, arithmetic execution, and sensitivity to answer format and prompting. These are multiple failure modes, not proof that a 3B model cannot learn statistics.

We compared original Llama 3.2 3B Instruct and the selected v0.5 adapter on 24 already-exposed v0.6 questions: indices 0,1,4,5 from each of six topics. No training, no teacher calls, no new held-out accuracy claim. v0.6 was not retested. [Protocol](STATS_DIAGNOSTIC_V0_7_PROTOCOL.md).

## Results

Counts below use the fixed strict parser. The manual-format column adds only unmistakably correct completed numerical answers rejected for their presentation; it does not count unfinished calculations.

| Condition | Baseline strict | v0.5 strict | Baseline after format review | v0.5 after format review |
|---|---:|---:|---:|---:|
| Original MC, four rotations | 29/96 | 41/96 | 29/96 | 41/96 |
| Supplied correct value → option, four rotations | 66/96 | 96/96 | 66/96 | 96/96 |
| No choices, direct value | 3/24 | 3/24 | 3/24 | 4/24 |
| No choices, short calculation then value | 2/24 | 7/24 | 3/24 | 7/24 |
| Correct rule supplied, direct value | 2/24 | 4/24 | 2/24 | 5/24 |
| Correct substituted expression supplied | 3/24 | 5/24 | 3/24 | 6/24 |

MC versus free-response percentages are not interchangeable measures: MC offers candidate values and chance success. Mapping intentionally gives away the answer. It tests a simpler task, not mathematical reasoning.

**Length limitation:** short-calculation responses hit the 192-token cap in 16/24 baseline and 6/24 v0.5 cases. Some had useful/correct intermediate work. Their strict scores are not reliable estimates of unconstrained calculation performance. The other conditions had zero length-limit hits. Format-invalid counts for baseline / v0.5 were: direct 0/4, steps 15/6, guided 0/4, arithmetic 3/5. Hitting the cap and being invalid are different flags.

The four format-only corrections are fully enumerated in the JSON audit:
- Baseline Poisson index 05, steps: completed final sentence gives Var(S)=32.
- v0.5 Poisson index 01, direct: standalone Var(S)=16.
- v0.5 uniform index 01, guided and arithmetic: <12.0>.

No arbitrary intermediate number was extracted to inflate final-answer accuracy. Correct final values also do not certify that every explanation statement is valid.

## Concrete observed failures

1. **Variance scaling:** for Poisson X with mean 4 and S=2X+5, v0.5's short calculation says to multiply variance by 2, yielding 8; the correct value is 16. It answers the supplied expression 2^2*4 correctly. In direct mode it also gives 16, demonstrating prompt-dependent inconsistency rather than a uniformly missing fact.
2. **Arithmetic after an appropriate setup:** for exactly one detection with miss probability 3/10, v0.5 writes (7/10)*(3/10)+(3/10)*(7/10), then evaluates it as 21/100 instead of 42/100. The event decomposition can be right while execution fails.
3. **Even supplied arithmetic is unreliable in direct-answer mode:** v0.5 answers (6+12)/2 with 18 and (14+28)/2 with 42. This cannot be explained by not knowing the statistical rule. It is not a claim that every alternative arithmetic prompt would fail.
4. **Confidence-interval concept failure:** for interval [30,42] with sample size multiplied by nine, v0.5's steps multiply the upper endpoint by nine and give 378. The correct result is 38: center 36 plus half-width 6/3. The same wrong endpoint rule appears in all four selected confidence questions.
5. **Choice dependence:** the four uniform questions are correct on all 16 MC rotations, but direct no-choice answers are wrong on all four, even after format review. One short-calculation answer is correct. MC success alone therefore overstates evidence of reliable independent calculation.
6. **Some real improvements:** v0.5 fixes the two Poisson arrival-count explanations relative to baseline and completes several probability calculations correctly. The trained model is neither uniformly worse nor uniformly incapable.

## What this supports

A basic inability to associate a known correct value with A/B/C/D is not the main remaining explanation for v0.5 on this sample: supplied-value mapping is 96/96. However, ordinary question answering remains position-sensitive, and simplifying the task can itself change behavior.

The evidence supports a mixture of weak rule application, unreliable arithmetic and prompt dependence. Training improved MC and supplied-value matching, but did not establish dependable open numerical solving. The current data and objectives may be teaching recognition and response patterns more effectively than independent solution execution; that is an interpretation, not a proven causal mechanism.

A targeted next experiment should use audited short numerical solution steps and error-specific contrasts (variance scaling, event combination, interval center/half-width), rather than simply appending more long explanations. First define a concise output format and a sufficient generation budget. Keep numerical solution and option selection separate in the evaluation. Any new training needs a newly frozen evaluation because this set is now diagnostic development material. No new training was started in this run.

## Verification and preservation

- All **576 responses** retained in [machine-readable results](STATS_DIAGNOSTIC_V0_7_RESULTS.json), with exact prompts, references, raw text, token counts, strict scores and format-review exceptions.
- All **192 numerical-condition raw responses** independently read for this report.
- Both models reproduce **96/96 original MC raw responses** from the v0.6 run.
- Three scoring tests passed; all 24 supplied arithmetic expressions were independently evaluated with exact rational arithmetic and matched the frozen answers.
- Completed-run verifier checked task coverage, scores, token flags, ZIP CRC and seven manifest entries.
- Same pinned base revision and v0.5 adapter hash; NF4, one visible Tesla T4, greedy decoding. Runtime versions are in the JSON. No full-precision control or alternate model was tested, so this does not isolate model size, quantization or architecture.
- Kaggle **version 12 (347608100)** confirmed Successful with saved diagnostic ZIP present. Model weights remain in version 9; this diagnostic archive contains results, not a replacement model.
- Archive: 3beethoven_stats_diagnostic_v0_7.zip, **32,976 bytes**.
- SHA-256: `3bf4a8d136fbf27d2df093708a83c31bda5c9161a24bafdf8c0b9d549216527d`.
- **Zero teacher calls and zero added teacher API cost.** GPU session stopped after saving.

For recovery, download pinned notebook version 12 and validate the ZIP hash before extracting to a fresh /kaggle/working/3beethoven_stats_diagnostic_v0_7 directory. Run scripts/verify_stats_diagnostic_v0_7.py from the repository to inspect saved results without training or teacher requests. To rerun inference, remove any existing input for this same notebook and mount version 9 for the checked v0.5 adapter; do not use Run All on the historical notebook.

These 24 questions belong to six parameterized families. Rotations are correlated, there is one seed and one quantization setup, prompt interventions are not pure causal controls, and the short-calculation budget censored many outputs. The defensible conclusion is a diagnosis of this checkpoint under these conditions, not a general ceiling on 3B models.
