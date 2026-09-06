# v0.8 compact-solution and length-control diagnostic

## Correction to the earlier diagnosis

**The v0.7 direct-answer arithmetic result understated what these models can do with a different prompt.** With a compact Formula / Calculation / Answer prompt, v0.5's supplied-expression accuracy rises from 5/24 to 18/24 under the same strict parser, or from 6/24 to 20/24 after explicit format review. This weakens a general "cannot calculate" explanation. It does not eliminate arithmetic failures or establish independent statistical problem solving.

## Fixed comparison

Same 24 exposed v0.7 questions, same pinned baseline and v0.5 adapter. No training or teacher calls. Compact prompts allow 256 tokens. Original step prompts allow 512 tokens: reuse the 26 earlier completed step answers and rerun only the 22 capped answers, without changing their prompts. There are 144 condition records, of which 118 are newly generated.

| Condition | Baseline strict | v0.5 strict | Baseline after format review | v0.5 after format review |
|---|---:|---:|---:|---:|
| Compact complete problem | 3/24 | 8/24 | 4/24 | 8/24 |
| Compact supplied arithmetic | 14/24 | 18/24 | 18/24 | 20/24 |
| Original step prompt, larger budget | 8/24 | 10/24 | 9/24 | 10/24 |

No v0.8 answer hit its generation cap. The v0.7 strict step scores were baseline 2/24 and v0.5 7/24; the larger budget recovers six and three correct final answers respectively. The compact full-problem score is not better than the longer original-step prompt in this sample. Compact format is therefore a candidate training/output format, not an already demonstrated universally superior method.

All six extended v0.5 outputs preserve the previous decoded prefix exactly. Fifteen of sixteen baseline extensions do too; the remaining earlier prefix ends with a Unicode replacement character from truncating the integral symbol. The meaningful text agrees up to that broken character. This is recorded explicitly rather than reported as 22/22 literal string matches.

The [raw JSON](STATS_DIAGNOSTIC_V0_8_RESULTS.json) lists all eight format-only corrections. These accept completed numerical conclusions, not arbitrary intermediate values. Strict results remain unchanged.

## What still fails

- Given the expression, v0.5 solves all four expectation and all four uniform arithmetic examples correctly.
- Given the complete question, it still fails all four expectation and confidence-interval questions with the compact prompt.
- On all four uniform questions, compact no-choice solving is now correct, whereas v0.7's direct-answer condition was wrong on all four. This is strong prompt dependence, not a universal inability to solve uniform problems.
- The model can output a correct final number while also naming a false or irrelevant formula. For example, a correct Poisson result appears alongside "Var(X)=lambda^2". Final-answer scoring alone does not certify explanation quality.
- Arithmetic errors remain: one compact supplied binomial expression drops a factor of ten, another drops a factor of three.

The next training intervention therefore targets **translation from problem to formula and reliable substituted calculation**, using independently audited concise teacher solutions. It must still be evaluated on separately frozen questions. These diagnostics are already exposed development material.

## Preservation and limits

Archive: 3beethoven_stats_diagnostic_v0_8.zip, 19,500 bytes, SHA-256 `e0662dd2179965909f1463fa74888b4443289ff647cb705ed813d4e6bc296107`. The working archive is included with the subsequent saved experiment output; its persistent Kaggle version is recorded in the recovery guide after save verification.

Both formats and generation budgets changed in the compact-versus-direct comparison. This is not a pure causal test of "thinking", and no private internal computation is inferred from the text. Question families, quantization, model and seed are narrow. The supplied-expression condition gives away the hard translation step and is not comparable to unaided teacher or MC scores.
