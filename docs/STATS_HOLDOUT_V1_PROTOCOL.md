# Frozen statistics holdout v1

Checkpoint: evaluate the existing v0.3 adapter, vanilla 3B and Llama 70B on 60
new questions, analyze errors, and preserve results. No training is permitted.

The user explicitly authorized up to **60 additional teacher API attempts**
for this checkpoint. Its separate write-ahead ledger has a hard cap of 60;
the previous run's 94 attempts and ledger remain unchanged. Requests are cached
and never automatically retried after ambiguous failures. Invalid model answers
are counted as wrong, not repaired. Existing provider price and byte limits
remain in force. No new credentials are required or saved.

Freeze all questions, independently specified gold answers, their SHA-256, the
adapter SHA-256 and the base-model revision before the first evaluated request.
There are six concepts, ten questions each, with A/B/C/D each 15. The question
file is evaluation-only and must never be used as student training targets.
Numeric labels have separate arithmetic tests; all questions were reviewed for
ambiguity before inference. Exact overlap with both prior curriculum and exam
is prohibited; semantic overlap is possible and is disclosed.

This is an **internally authored post-training holdout**, not an externally
authored or independently certified blind benchmark. Its author knows the old
training topics and results. The tested models do not receive answer keys or
reference rationales. Questions include new transfer tasks as well as familiar
principles; do not compare its aggregate score directly against the easier old
24-question development exam.

All three models receive the same v0.3 primary letter-only user prompt and a
16-token output limit. The same strict leading-letter parser and scoring are
used, with raw responses retained. Student inference is greedy, on one T4,
with the same pinned NF4 base revision; the baseline runs before loading the
unchanged saved adapter. The teacher uses temperature zero and its already
configured model. One serial teacher thread runs alongside GPU inference.

Report 60-question accuracy, invalid outputs, six per-concept breakdowns,
paired wrong-to-right/right-to-wrong counts, complete question-level outcomes,
API attempts and response-reported costs. Do not tune prompts, edit gold
answers, or retrain after seeing outcomes. Any later correction requires a
separately documented benchmark revision, not silently overwritten results.

Preserve a separate small ZIP with all question data, raw outputs, paid caches
without authorization headers, protocol, source snapshots, summary and file
manifest. The adapter remains in the already saved v0.3 archive. Save Kaggle
with output explicitly enabled, verify the output listing, and put code and
results on GitHub. If browser downloads remain unavailable, retain the ZIP on
Kaggle for the user's explicitly approved manual retrieval.
