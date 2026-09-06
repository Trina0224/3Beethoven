# v0.5 expanded curriculum protocol

User selected curriculum expansion after v0.4 failed to improve the new transfer probe. Retain Llama-3.3-70B-Instruct as teacher and Llama-3.2-3B-Instruct as student. No ChatGPT answer explanations become student targets.

## Frozen curriculum

| Split | Questions | Task families | Purpose |
|---|---:|---:|---|
| Training | 180 | 18 (three per topic) | New validated teacher explanations |
| Validation | 24 | 6 (one per topic) | Checkpoint selection |
| Test | 36 | 6 (one per topic) | Frozen transfer assessment |

Six topics: Poisson counts, expectation, uniform distributions, Type I errors, Type II errors, and confidence intervals. Families are disjoint between the three splits within this run; related mathematical principles intentionally transfer. Each family contains parameterized examples, not independent reasoning skills. The author knows the earlier diagnostic results; some mathematical forms resemble previous exposed probes. This is an internally authored test of the new training curriculum, not an external blind benchmark.

The generator uses exact rational arithmetic, balanced A/B/C/D positions, distinct numeric option values, no duplicate questions, and checks exact overlap with earlier curricula/probes. Freeze all concrete questions before teacher generation and student training. Mathematical reference notes are validation inputs, not student explanation targets.

## Teacher generation and budget

Generate 204 training/validation records. Check the answer letter against the exact reference, then ask the same Llama teacher to review the explanation and misconception. Same-teacher review can share errors and is not independent proof. Preserve all raw generations, reviews, rejected attempts, and whether a repair was reference-conditioned. At most three attempts per question.

Use up to four concurrent requests, a durable pre-request ledger, unique cache tags, and a thread-safe reservation cap of 600 attempts. No automatic retries of unresolved network requests. Prompt text is capped at 4,000 UTF-8 bytes and output at 400 tokens per request. Provider ceilings are $1/M input and $2/M output tokens, with zero per-request charge; this gives an approximate conservative text-token bound below $3 for 600 maximum-size calls, before any outside billing adjustments. Record actual response-reported costs and do not present them as invoices or account balances. Expected successful requests: 408 generation/review calls plus 36 original-order teacher test calls, before repairs.

Teacher test responses remain separate from training records. Report teacher performance without treating it as ground truth or silently excluding its errors. Preserve the generated corpus before GPU training.

## Student recipe and evaluation

Return to the v0.3 recipe: original-order letter target plus original-order teacher explanation per question, without v0.4's four-to-one format weighting. Train a fresh adapter from base revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`, with NF4, LoRA rank16/alpha32/dropout0.05, seed226, learning rate5e-5, effective batch8, three epochs, and validation-loss checkpoint selection. 360 training sequences imply 135 optimizer steps; this increases compute as well as curriculum size compared with v0.3's 36 steps. Do not claim pure causal isolation of diversity versus training amount.

Compare baseline, saved v0.3 and new v0.5 on the 36 new questions and the exposed 60-question regression set, all with four cyclic rotations and 16-token greedy outputs. This is 384 responses per student model. The teacher has only an original-order comparison, not a rotation score. Primary metrics: new-set four-rotation mean accuracy and all-four-correct count; also report semantic consistency, positions, topics, and regressions. Questions remain clustered within six test families; rotations are not independent observations. No configuration search on test results.

Save corpus, source snapshots, protocols, raw responses, logs, adapter and checksummed archives. GitHub receives code and text artifacts; Kaggle retains model archives. Do not replace the preferred v0.3 checkpoint unless the new evidence warrants it.
