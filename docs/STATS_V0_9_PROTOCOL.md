# v0.9 targeted concise-solution protocol

Frozen before revised teacher generation or student evaluation. This follows v0.7's observed rule/application/arithmetic failures and v0.8's compact-output diagnostic.

## Data and interpretation
180 training questions, 24 validation questions, 48 test questions across six statistics topics. Parameter tuples are disjoint across splits; split-specific narratives also differ. Test skills are deliberately covered in training. This evaluates new instances of taught skills, NOT unseen task-family transfer. Curriculum alignment, target format and data all differ from v0.5; it is NOT an isolated teaching-style ablation.

The complete frozen data are in STATS_V0_9_FROZEN_QUESTIONS.json, SHA-256 (canonical digest) 96987f7fddcc6c74f80ff1c3d27f5eb314a9732e263ff62bd8d4ffd72fb73627. Old v0.6/v0.7 questions are exposed diagnostics and never a new success test.

## Teacher
Llama 3.3 70B Instruct only. Each train/validation target has three short lines: Formula, Calculation, Answer. First teacher attempt solves independently; failed numeric checks receive exact reference feedback. Preserve reference-conditioning flags and every paid response. Check every numerical equality exactly; independently read all rule prose before training. No ChatGPT mathematical text substituted into assistant targets.

Durable cap: 500 teacher requests including retries and later teacher test calls; max 400 output tokens, existing input-byte and provider-price limits. Never reset ledger. Record reported cost, not guessed account balance. If a mathematical claim fails audit, request a Llama correction and preserve both responses.

## Training
Fresh pinned Llama 3.2 3B Instruct base, NF4, seed 226, LoRA r16/alpha32/dropout .05, LR 5e-5, effective batch 8, 3 epochs / 135 optimizer steps. 360 sequences: one multiple-choice letter target plus one concise no-choice numerical solution per training question. Validation: same two modes, 48 sequences. Max sequence length 768, no silent truncation. Select minimum validation loss, never test-score selection. Save/reload selected adapter for evaluation.

## Evaluation
Baseline, v0.5 and new student: same new 48 questions under four MC rotations (192/model) and concise no-choice numeric prompt (48/model), greedy decoding. MC budget 16 tokens, numeric 256. Preserve all output and caps; independently audit invalid outputs and calculate any format-only sensitivity separately. v0.9 also runs old exposed 60×4 as a regression check; old baseline/v0.5 figures are historical references, not newly generated old-set controls.

Primary inherited targets: new MC accuracy >=60%, >=2×same-run baseline, and >=24/48 questions correct on all rotations. Report each separately, without hiding misses. Numeric solution score is a distinct secondary metric, not interchangeable with MC. Teacher test (original-order MC only) must be recorded separately and excluded from training/audit targets.

## Preservation
Code, corpus, audit and raw evaluation in GitHub; adapter/ZIP in Kaggle. Preserve negative results. Stop GPU after verified saved output. A better result on this aligned curriculum does not retroactively fix v0.6's failure on its different test.

## Pre-training validation amendment

Teacher responses sometimes contain correct symbolic derivations followed by a valid numerical chain. The renderer may remove only the leading symbolic prefix and make implicit numerical multiplication explicit. It retains the entire numerical suffix and checks every equality; it must never discard a false numerical intermediate. Original paid responses remain in the cache, and records mark numeric-suffix normalization. This is deterministic formatting/extraction of Llama text, not replacement with newly authored mathematical targets. Bounded reference-copy repair attempts are allowed within the same 500-call ledger; their reference-conditioning is preserved.

## Independent prose-audit amendment, before training

All 204 worked calculations were read after numeric validation. The audit found vague topic labels, unqualified inclusive-union equations used for exactly-one events, and a false conditional-expectation rule. Replace every Formula line with one of seven shared Llama-generated rules (two Poisson cases and five other topics), independently checked before assembly. Keep each example's validated Llama numerical calculation and answer unchanged; preserve original rule text and separate rule-cache provenance. Three hard numerical repairs explicitly reproduced a supplied verified JSON object; these are reference-formatting repairs, not independent teacher solutions. Frozen questions, split membership, success criteria and training schedule stay unchanged.
