# Execution Handoff — Step180 Delivered

## Completed work

- One 1,440-example training trajectory: 720 new examples plus 720 materialized replay examples; 180 optimizer updates.
- Final evaluation on 180 reserved questions: original 3B 45/180; step180 170/180.
- Student format: 180/180 strict one-line expressions; no pending answers.
- All ten student failures were inspected against the questions and reference expressions; they are actual formulation errors.
- Weights are published on [Hugging Face](https://huggingface.co/kozakurayuki/3Beethoven-step180); raw results are in [the evaluation ZIP](../final_blind_eval_results.zip).

## Reproduction identity

- Base: `meta-llama/Llama-3.2-3B-Instruct`
- Base revision: `0cb88a4f764b7a12671c53f0838cd831a0843b95`
- Adapter: `kozakurayuki/3Beethoven-step180`
- Adapter revision: `682d5555b0e6115135ac7b9b5d718d2abef186de`
- Adapter SHA-256: `ff89a22f097e4db457dab287042ae36520ec6b9b36f26e5ed11fe42337c04f23`

Load this adapter directly on the pinned base. Do not stack it on v15. Use the base tokenizer, not the old tokenizer configuration from the Kaggle archive.

Use the [Colab instructions](STATS_DIVERSE_FINAL_BLIND_COLAB_HANDOFF.md). Both arms use the same base tokenizer, prompts, greedy decoding, token budget, grader, and NF4 loading configuration. The adapter archive's old `TokenizersBackend` configuration is incompatible with the tested Transformers runtime; the evaluator now explicitly loads the pinned base tokenizer.

## Interpretation

The earlier v15-based frozen selection protocol remains `no_checkpoint_passed`: it required zero paired losses and had documented execution deviations. The owner subsequently requested a separate final comparison against the unmodified 3B model. This supports a bounded project success claim; it does not retroactively pass the earlier protocol or establish zero regressions.

This is evidence for the tested statistics-expression task, not general language ability, arbitrary mathematical reasoning, or isolated teacher-transfer causality. The run combines corrected synthetic supervision and historical replay; no ablation isolates their individual effects. The final set is now exposed and must not be reused as an untouched test for future model selection.

[Preserved Traditional Chinese version](STATS_EXECUTION_HANDOFF.zh-TW.md)
