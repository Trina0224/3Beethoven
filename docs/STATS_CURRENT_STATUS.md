# Current Status — Evaluation Complete

The current successful bounded-task artifact is **step180**, available on [Hugging Face](https://huggingface.co/kozakurayuki/3Beethoven-step180). No training or evaluation is currently running.

| Metric | Original 3B Instruct | Step180 adapter |
|---|---:|---:|
| Automatically correct | 45/180 (25.0%) | **170/180 (94.4%)** |
| Strict one-line format | 26/180 | **180/180** |
| Pending review | 66 | **0** |

Paired outcomes: 127 wrong-to-right, 2 right-to-wrong, 43 both correct, and 8 neither correct. Net gain: 125 questions (+69.4 percentage points). Category totals improved in 17 of 18 categories and tied in one; none declined under the automatic scoring rule. Even crediting all 66 baseline pending answers would bring that baseline to only 111/180, below 170/180. This is a sensitivity bound, not a completed semantic review of those 66 answers.

## Artifact identity

- Base: `meta-llama/Llama-3.2-3B-Instruct`
- Base revision: `0cb88a4f764b7a12671c53f0838cd831a0843b95`
- Adapter: `kozakurayuki/3Beethoven-step180`
- Adapter revision: `682d5555b0e6115135ac7b9b5d718d2abef186de`
- Adapter SHA-256: `ff89a22f097e4db457dab287042ae36520ec6b9b36f26e5ed11fe42337c04f23`

Load this adapter directly on the pinned base. Do not stack it on v15. Use the base tokenizer, not the old tokenizer configuration from the Kaggle archive.

## Interpretation

The earlier v15-based frozen selection protocol remains `no_checkpoint_passed`: it required zero paired losses and had documented execution deviations. The owner subsequently requested a separate final comparison against the unmodified 3B model. This supports a bounded project success claim; it does not retroactively pass the earlier protocol or establish zero regressions.

This is evidence for the tested statistics-expression task, not general language ability, arbitrary mathematical reasoning, or isolated teacher-transfer causality. The run combines corrected synthetic supervision and historical replay; no ablation isolates their individual effects. The final set is now exposed and must not be reused as an untouched test for future model selection.

[Final report](STATS_DIVERSE_FINAL_BLIND_RESULTS.md) · [Execution handoff](STATS_EXECUTION_HANDOFF.md)

[Preserved Traditional Chinese version](STATS_CURRENT_STATUS.zh-TW.md)
