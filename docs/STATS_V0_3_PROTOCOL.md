# Statistics v0.3 repair experiment

## What this run can and cannot establish

This is a repaired statistics pilot within the broader 3Beethoven project. It
uses the **same 24-question v0.2 development benchmark** for a freshly measured
vanilla 3B baseline, trained 3B, and Llama 3.3 70B teacher. That benchmark was
already exposed in previous work. It is not a new blind test and must not be
used to claim broad or statistically established improvements.

The reported v0.1 62.5% was on a different 16-question benchmark. Comparing it
directly with v0.2's 41.67% does not establish regression.

## Verified engineering issues

- v0.2 did not measure a same-exam vanilla baseline.
- Training requested an answer and explanation, whereas evaluation requested
  only a letter and allowed four output tokens. This is a potential confound,
  not a proven explanation of the low score.
- The v0.2 resume patch used a double-escaped `\\W+` normalization pattern,
  which does not perform the intended punctuation/whitespace normalization.
- The v0.2 Kaggle quick-save viewer retained the early Secret error, and its
  Output page did not expose the subsequently trained adapter/corpus. The
  restarted working directory no longer contained those artifacts. The earlier
  preservation claim was not adequately verified.
- In the first v0.3 generation attempt, conflicting JSON and prose instructions
  produced two correct plain-text answers. Their cached responses are retained
  and recovered without new calls. Later generation requests only JSON.

## Fixed protocol

- Teacher: `meta-llama/llama-3.3-70b-instruct` (70B, not 30B).
- Student: `meta-llama/Llama-3.2-3B-Instruct`, NF4 QLoRA, one visible T4.
- Curriculum: 60 deterministic, checkable multiple-choice questions; six
  concepts x 10 questions; answer positions A/B/C/D each 15.
- The program supplies questions, options, and independently computed labels.
  The teacher independently answers and supplies explanations. Its answer must
  agree with the reference. This is **teacher-explanation response distillation**,
  not wholly teacher-generated questions or logit-based distillation.
- If an independently generated answer fails, a retry includes the program's
  reference calculation and option mapping. Such responses are marked
  `reference_conditioned`; they are corrected explanation synthesis, not
  evidence that the teacher solved the question unaided. At most three attempts
  per question, still subject to the persistent 120-attempt total ceiling.
- Teacher evaluation: separate cache, never added to the training dataset.
  Stop generation if teacher accuracy is below 80% on the development set.
- Check exact normalized question duplicates and exact evaluation overlap.
  This is not a guarantee against semantic/template similarity.
- Split by original question before creating format variants: 48 train / 12
  validation (8 / 2 per concept). Each training question has a letter-only and
  explanatory example, giving 96 training sequences. Format variants never
  cross the split.
- A pre-training review of all 60 explanations also identified five content
  issues (an unjustified Poisson uniqueness claim, a valid expectation identity
  mislabeled as a mistake, and three confidence/precision confusions).
  `review_stats_v0_3.py` asks the same Llama teacher to revise those explanations
  using explicit review feedback. Original records and all paid responses are
  retained. Review feedback is not itself a student target. These five calls
  share the original 120-attempt ceiling; seven final records in total are
  reference-conditioned (two answer corrections and five content revisions).
- Seed 226; 3 epochs; learning rate 5e-5; LoRA r=16, alpha=32, dropout=.05;
  batch size 1, gradient accumulation 8. Choose the checkpoint by validation
  loss, not benchmark accuracy.
- Preserve and assert the tokenizer's chat-template prefix; refuse truncation
  or examples with no supervised tokens.

## Predeclared evaluation

All three student modes are evaluated both before and after training:

| Mode | Prompt | Token limit | Role |
| --- | --- | --- | --- |
| `legacy4` | Original v0.2 letter-only prompt | 4 | Compatibility diagnostic |
| `letter16` | Same letter-only prompt | 16 | Primary paired comparison |
| `explain64` | Answer plus short explanation | 64 | Format sensitivity diagnostic |

Store every raw response, parsed answer, expected answer, token count, invalid
response, and token-limit flag. The parser accepts only a leading letter or an
explicit leading `Answer:` marker, never an arbitrary letter found in prose.
Report wrong-to-right and right-to-wrong transitions, not just aggregate scores.

Several things change from v0.2 (data, formats, learning rate, validation split).
This repair run does not isolate the causal contribution of each change. A
future controlled ablation and a separately authored blind benchmark are needed.

## Budget and persistence

The user's ceiling is 120 new API attempts across evaluation, generation,
validation retries, and resumes. A write-ahead JSONL ledger counts each attempt
before transmission. Completed calls have cached requests and responses;
ambiguous in-flight failures stop rather than automatically rebill. HTTP failures,
including authentication, credit, and rate limits, stop the run.

Each request is limited to 400 output tokens and 4,000 input-content bytes;
provider prices are capped at $1/million input and $2/million output tokens,
with no per-request fee. Actual response-reported cost is recorded separately
from call counts. These records are not a substitute for the account invoice.

Reference: [OpenRouter provider price filters](https://openrouter.ai/docs/guides/routing/provider-selection#max-price)
and [usage accounting](https://openrouter.ai/docs/cookbook/administration/usage-accounting).

The run writes a ZIP after generation and after training. It contains data,
API caches without authorization headers, split identifiers, raw evaluations,
training logs, source snapshots, environment versions, and the adapter when
available. Per-file SHA-256 checks and ZIP CRC verification are included.
Kaggle's `Successful` label alone is **not** proof that these files were saved.
Verify an actual downloaded archive before ending the interactive session.

The Quick Save advanced setting was observed as **Never save output**. Use
**Save output for this version when creating a Quick Save** explicitly. During
this rerun, version 2's Output page listed the 142,636-byte initial corpus ZIP,
but browser download events timed out. A file listing establishes saved output
availability, not successful local download or end-to-end byte verification.

## Commands

```bash
python -m unittest discover -s scripts -p 'test_stats_v0_3.py' -v
CUDA_VISIBLE_DEVICES=0 python -u scripts/flight_run_stats_v0_3.py --stage generate --max-calls 120
CUDA_VISIBLE_DEVICES=0 python -u scripts/flight_run_stats_v0_3.py --stage train --max-calls 120
```

Do not run the old v0.1/v0.2 generation cells when resuming this experiment.
