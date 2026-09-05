# v0.4 position repair: frozen protocol

This bounded experiment follows the v0.3 option-rotation diagnostic. It uses no additional teacher calls and does not add new teacher knowledge.

- Start from the same pinned Llama-3.2-3B-Instruct base, not the v0.3 adapter.
- Reuse the 60 validated Llama-3.3-70B training records from Kaggle version 5 and the exact 48/12 question split. Check the corpus against deterministic references and the saved adapter hash before use.
- Each training question yields four cyclically rotated letter-only targets, with semantic labels mapped deterministically from the accepted teacher answer, plus one unchanged original-order explanation target. Do not rewrite teacher explanations or move them to reordered prompts.
- 240 training sequences and 60 validation sequences. Split by original question before augmentation.
- Same NF4 base, LoRA rank 16, alpha 32, dropout 0.05, learning rate 5e-5, seed 226, effective batch 8, warmup 2, cosine schedule. Exactly 36 optimizer steps, matching the prior step budget. This is about 1.2 augmented-data epochs rather than three old-data epochs.
- Validation and checkpoint selection every 12 steps using validation loss only. Reload the saved adapter for final evaluation.
- This changes both augmentation and the letter/explanation mixture. Equal optimizer steps do not mean equal supervised tokens. It is not a pure causal isolation of option rotation.

## Evaluation frozen before training

Evaluate the baseline, saved v0.3 adapter, and new v0.4 adapter with greedy decoding and a 16-token cap:

1. New 24-item evaluation-only transfer probe, six parameterized task families with four items each, in `scripts/stats_holdout_v2.py`. These internally authored questions combine existing statistical principles; they are not externally blind or 24 independent task families.
2. Previously exposed 60-item holdout, retained only as a regression diagnostic.
3. Four cyclic rotations per item, not all 24 permutations. 336 responses per model, 1,008 total.

Primary assessment: four-rotation mean accuracy and all-four-correct question counts on the new probe; report semantic consistency, old-set regressions, position and topic breakdowns regardless of outcome. Do not tune on test outcomes or rerun configurations until a favorable score appears. No teacher comparison is added to the new probe.

Freeze the protocol, scripts, and question generator in GitHub before running. Store hashes, split, exact expanded training targets, raw responses, package versions, training logs, saved adapter and an integrity-checked archive. This experiment only addresses response/position behavior using existing knowledge; broader curriculum expansion remains separate.
