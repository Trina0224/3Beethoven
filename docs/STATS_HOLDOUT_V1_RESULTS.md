# Statistics holdout v1 — completed checkpoint

## Outcome

The unchanged distilled 3B scored **35/60 (58.33%)**, versus **21/60 (35.00%)**
for vanilla 3B and **52/60 (86.67%)** for the Llama 70B teacher. The paired
student difference is **+14 answers / +23.33 percentage points**: 25 answers
changed from wrong to right and 11 changed from right to wrong.

| Model | Correct | Accuracy | Invalid outputs |
| --- | ---: | ---: | ---: |
| Vanilla Llama 3.2 3B | 21/60 | 35.00% | 0 |
| Distilled Llama 3.2 3B | 35/60 | 58.33% | 0 |
| Llama 3.3 70B teacher | 52/60 | 86.67% | 0 |

The models received identical individual-question prompts and 16-token limits.
No student response reached the limit. There was no training or post-result
prompt tuning. These new scores must not be directly compared to the old
24-question development score of 75%, because the questions and difficulty
changed. The valid comparison is between models on this same new set.

## Where the gain is strongest

Each category contains ten questions.

| Category | Vanilla | Distilled | Teacher |
| --- | ---: | ---: | ---: |
| Poisson | 4 | 6 | 10 |
| Expectation | 3 | 4 | 7 |
| Uniform distributions | 3 | 4 | 7 |
| Type I error | 3 | 7 | 9 |
| Type II error | 5 | 9 | 10 |
| Confidence intervals | 3 | 5 | 9 |

The clearest gains are in hypothesis-test error concepts. Arithmetic,
transformed/conditional expectations, variance scaling, and interval-width
calculations remain weak. Example student misses include:

- `holdout_05`: incorrectly requires independence for a linear expectation.
- `holdout_08`: incorrectly requires identical distributions to subtract means.
- `holdout_03`: uses 162 instead of 2 for the variance of an independent sample mean.
- `holdout_28`: misses the squared multiplier in Var(2X).
- `holdout_21`, `holdout_58`: misses how interval width scales with sample size or standard deviation.

These examples illustrate observed choices, not directly observed internal
reasoning: the evaluation only requested answer letters.

## Important answer-position warning

Correct answers are balanced, with 15 each at A/B/C/D, but model choices are
not. Vanilla chose A on 49/60 questions; the distilled model chose A only 5
times (B=18, C=18, D=19). Ten of the eleven regressions have gold answer A.

| Gold answer position | Questions | Vanilla correct | Distilled correct | Teacher correct |
| --- | ---: | ---: | ---: | ---: |
| A | 15 | 14 | 4 | 14 |
| B | 15 | 6 | 10 | 13 |
| C | 15 | 0 | 9 | 13 |
| D | 15 | 1 | 12 | 12 |

This prevents attributing the entire score gain to improved statistical
knowledge. Reduced vanilla A bias and a new relative avoidance of A may be
part of the effect; the exact causal contribution is unmeasured. The most
useful next diagnostic is a **predeclared option-permutation robustness test**
on the two student models, not immediate additional training. It would not
require teacher API calls, but has not been run in this checkpoint.

## Teacher limitations

All eight teacher misses are preserved in the raw results:

| Question | Error |
| --- | --- |
| 05 | Linear combination of two means |
| 12 | Conditional remaining uniform waiting time |
| 15 | Expected false rejections across tests |
| 22 | Total confidence-interval width versus half-width |
| 31 | Expectation after expanding a square |
| 36 | Mixture mean |
| 37 | Uniform tail probability |
| 39 | Uniform variance |

The distilled model answered teacher misses 12, 36 and 39 correctly, but this
does not negate the teacher's substantially higher overall score. No teacher
answer was used to revise the gold key, and no paid correction calls were made.

## Freeze, provenance and limits

- Questions/protocol were committed before inference (freeze sequence ending
  `2e692474bbf01fef2f4133d6a2b2aef849d312d2`).
- Benchmark SHA-256: `9ab52132b6070eb69281e884ce256322730b1debb58407d7ed26048856bba5c4`.
- Adapter SHA-256: `7c3dd4513bd4f9e98ae03b9788f60a5337689de20056936cf03f7dba02bed4cf`.
- Base revision: `0cb88a4f764b7a12671c53f0838cd831a0843b95`.
- Nineteen tests passed; numeric labels have separate arithmetic checks, and
  exact overlap with both old curriculum and development set was checked.
- This is an internally authored, post-training holdout, **not an externally
  authored or independently certified blind benchmark**. The author knew the
  old topics and results; semantic familiarity is possible despite exact
  deduplication. One seed and 60 questions do not establish broad competence.
- Once evaluated, these questions are exposed. Preserve them for diagnostics;
  do not convert them to student training targets or silently adjust the key.

## Budget and preservation

The new authorization was **up to 60 additional teacher attempts**. Exactly
60 attempts produced 60 responses; response-reported cost was **$0.00106343**,
with no missing cost fields. The previous 94-call ledger was not changed.
Together the two experiments have 154 calls and reported cost $0.007196965.
These are response-reported totals, not an independently verified account bill.

The separate holdout ZIP contains the benchmark, frozen protocol, all raw
responses, per-question outcomes, summary, paid-response cache without
authorization headers, API ledger, source snapshots and manifest. It does not
duplicate the already saved adapter.

- ZIP: `3beethoven_stats_holdout_v1.zip`
- Bytes: **95,612**
- SHA-256: `93914fa6ed9aa205a9832dcbb318665a2c83557c4b2e109ec938d3b359abfc3a`
- ZIP CRC and the size/SHA-256 of all **74 manifest-listed files** passed in Kaggle.
- Full question-level results and raw answers are also preserved in
  [STATS_HOLDOUT_V1_RESULTS.json](STATS_HOLDOUT_V1_RESULTS.json) on GitHub.

Kaggle **version 5**, scriptVersionId **347584475**, explicitly includes output.
Its [Output page](https://www.kaggle.com/code/trinashih/3beethoven-v0-2/output)
was inspected after saving and lists both the existing 92.73 MB model ZIP and
the new holdout ZIP, plus their directories. Browser downloads were not
retried, per the agreed manual-retrieval fallback. No local binary-archive
download is claimed; all 180 raw responses and 60 question outcomes are also
available directly on GitHub in the JSON linked above.

After this output verification, the interactive GPU session was stopped; the
menu changed from Stop session to Start session. Retrieve files from saved
version 5, not from a new empty interactive working directory.

See [the frozen protocol](STATS_HOLDOUT_V1_PROTOCOL.md).
