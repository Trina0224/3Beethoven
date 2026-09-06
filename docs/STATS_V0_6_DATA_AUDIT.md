# v0.6 paired teaching data audit

The final training method uses the same 180 v0.5 training questions and exact audited Llama answers. Each explanatory example adds a related question from the next family in the same topic, with its exact Llama explanation. The related question appears in the user prompt; assistant output adds only a mechanical heading and the original teacher explanation. Main letter targets and all 24 validation records remain byte-for-byte equivalent at the example level.

A deterministic family cycle (0→1→2→0), matching the numeric example index, creates 180 distinct contrasts. Every original training question appears once more as a contrasting example. Thus this is not merely a typography change: it adds paired context, answer exposure and supervised text. The question inventory, 360 sequence count, seed and optimizer schedule stay fixed. Target characters increase from 48,941 to 76,464; prompt characters increase from 85,268 to 118,816. Character counts are not token counts or FLOP measurements.

The initial abstract-card approach used 65 requests and exposed incorrect teacher generalizations, including dependence supposedly breaking expectation linearity, a finite width for a one-sided confidence interval, and incorrect dependent-event probability formulas. Same-teacher reviewers approved some erroneous claims. All abstract cards are excluded from final training, including mathematically correct ones. Their raw responses and earlier accepted/rejected files remain in the archive as research artifacts. The design amendment occurred before student baseline evaluation or training, with the same frozen 48-question test and goals.

All final mathematical answer explanations come from the previously audited Llama corpus. The pairing builder and trainer verify exact source equality for every record; tests additionally verify unchanged letter targets, unchanged validation examples, different-family/same-topic pairing, and correct prompt/target boundaries. No new assistant-written mathematical explanation becomes a student target.

| New request stage | Requests | Response-reported USD |
|---|---:|---:|
| Abstract cards and reviews, unused | 65 | 0.005038840 |
| Frozen teacher test, original order only | 48 | 0.000890590 |
| Total | 113 | 0.005929430 |

The 180-call durable cap was not reset. All 113 responses reported a cost. These are API usage figures, not an invoice or account balance. The teacher scored 30/48 (62.5%) on the frozen new-story test; this score is not comparable to previous tests with different questions, and the test was not replaced to improve it.

Pre-training corpus archive: Kaggle version 10, script version ID 347602173. Source data and exact pairing map: [preparation record](STATS_V0_6_PREPARATION.json), [v0.5 audited Llama data](STATS_V0_5_TEACHER_DATA.json). See the [amended protocol](STATS_V0_6_PROTOCOL.md) for the unchanged success goals and experimental limits.
