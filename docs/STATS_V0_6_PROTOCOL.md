# v0.6 teaching-method protocol

Keep v0.5's 180 training questions, 24 validation records, model, seed226, NF4, LoRA16/32/dropout0.05, 360 sequences, three epochs/135 steps, learning rate5e-5 and validation-loss selection. Start from the same fresh base. v0.5's saved adapter is the historical control and will be re-evaluated in the current runtime.

Change only training instructional text: 18 reusable Llama-generated rule/step cards, one per training family, precede existing Llama worked explanations. Explicit related-case contrasts replace the misconception paragraph. Labels, question wording, option order, and validation targets stay unchanged. No ChatGPT prose becomes a student target. Sequence counts and optimizer schedule are held fixed, but target token counts are not matched; this is an instructional-text intervention, not a pure causal isolation of each component.

Freeze 48 new story/parameter questions before generating cards; SHA256 9798cdb9ab8281f45dbb7cd14219c365a6231a6b8eebbe43a46ea858f30b51e8. Their mathematical principles overlap known tasks. This is internal story transfer, not an external blind or wholly unseen-skill benchmark. Retain the exposed old 60-question regression set. Compare baseline, v0.5 and v0.6 with four cyclic rotations, greedy decoding, 16-token outputs: 432 responses per model, 1296 total. Teacher takes only the original-order new test.

Predeclared goals: fresh-set average accuracy >=60%, at least twice the same-run baseline accuracy, and at least 24/48 fresh questions correct in all four rotations. Report each goal separately even if infeasible or missed. No test-based configuration search, replacement questions, or hidden repeated trials.

Teacher budget: at most 180 new calls across cards, reviews, retries and 48 teacher test responses, using a durable reservation ledger. Provider ceilings remain $1/M input and $2/M output with no per-request charge, input <=4000 UTF-8 bytes and output <=400 tokens. All new cards receive same-teacher review plus independent assistant reading before training; self-review is not proof. Preserve originals and caches. Do not overwrite v0.5.

Save code, frozen questions, cards, raw evaluations and report to GitHub; save model and checked archive to Kaggle. Stop GPU after saved output is confirmed.
