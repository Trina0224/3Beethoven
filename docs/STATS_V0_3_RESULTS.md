# Statistics v0.3 repair run — 2026-09-05

## Scope and budget

Teacher: `meta-llama/llama-3.3-70b-instruct` (70B, not 30B).
Student: `meta-llama/Llama-3.2-3B-Instruct`, NF4 QLoRA on one visible T4.
Kaggle allocated T4 x2; only GPU 0 was exposed to the training subprocess.

The persistent ledger records **94 API attempts and 94 responses**, inclusive
of teacher evaluation, generation retries, and content revisions. The approved
ceiling was 120. Response-reported cost totals **$0.006133535**, with no missing
cost fields; this is not an independently checked account invoice. Training and
student evaluation make no teacher API calls.

## Data checks

- 60 questions, six concepts x 10; A/B/C/D each 15.
- Program-generated questions, options, and independently calculated labels;
  Llama-generated explanations and misconception descriptions.
- Original-question split: 48 train / 12 validation; two response formats give
  96 training sequences and 24 validation sequences, with no cross-split twins.
- No exact normalized duplicates or exact development-question overlap.
- All 60 explanations were reviewed before training. Five were revised by the
  same Llama teacher: one unjustified Poisson uniqueness claim, one valid
  expectation identity incorrectly called a mistake, and three confusions
  between confidence/coverage and precision.
- Seven final records are reference-conditioned: two answer corrections and
  five content revisions. Original versions and paid-response caches remain
  in the run archive. Corrected explanations are not evidence of unaided
  teacher competence.

The teacher scored **21/24 (87.5%)** on the separate development evaluation.
Its three misses were linear-expectation questions. In rejected training
responses it also sometimes computed the correct value but selected a
different option, or contradicted its own arithmetic.

## Engineering findings

1. Both Kaggle Secrets, `HF_TOKEN` and `OPENROUTER_API_KEY`, were readable.
   No secret values were printed or committed.
2. Missing `bitsandbytes` was installed with approval.
3. The earlier v0.2 resume normalization used double-escaped `\\W+`; the
   repaired helper uses the intended `\W+` and has a regression test.
4. Transformers 5.0.0 returned `BatchEncoding` from `apply_chat_template`.
   The new boundary assertion caught the mismatch before any optimizer step.
   Explicit `return_dict=False` fixes token-list masking without removing the
   guard. The completed baseline cache was reused on resume.
5. Sixteen dependency-light unit tests passed locally and in Kaggle.
6. PEFT emitted a nonfatal unauthenticated config-lookup warning during saves;
   the model itself had loaded successfully with the configured HF token.
   Vocabulary was not modified.
7. Kaggle Quick Save was set to **Never save output**. The new saves explicitly
   select **Save output for this version**. A success label alone was not
   adequate evidence for the earlier v0.2 preservation claim.

## Same-exam comparison

The old v0.2 24-question development set is used for every row. It was already
exposed in previous work and is not a new blind test. The v0.1 62.5% score came
from a different 16-question test and is not directly comparable.

| Mode | Baseline | Distilled | Wrong to right | Right to wrong |
| --- | --- | --- | --- | --- |
| Legacy letter, 4 tokens | 8/24 (33.33%) | 18/24 (75.00%) | 12 | 2 |
| Primary letter, 16 tokens | 8/24 (33.33%) | 18/24 (75.00%) | 12 | 2 |
| Answer plus explanation, 64 tokens | 12/24 (50.00%) | 15/24 (62.50%) | 4 | 1 |

All baseline responses parsed successfully. The two letter-only modes produced
identical answers (22 A, 2 B); extending their limit did not change baseline
accuracy. The explanatory prompt improved baseline accuracy, demonstrating
prompt sensitivity on this set, not a causal explanation of the old v0.2 run.

All post-training responses also parsed successfully. The final letter-only
distribution was A=5, B=8, C=5, D=6. The explanatory prompt no longer had the
best accuracy after training, so explanatory output alone is not a general
remedy. Explanatory scores grade the leading answer, not the quality or full
correctness of the reasoning; 64-token outputs can be truncated.

Training completed 36 optimizer steps / three epochs in about 190 seconds.
Training loss was 0.6830482880. Validation losses decreased from approximately
0.5122 to 0.3683 to **0.3449130356**; checkpoint 36 was selected by validation
loss, not development-test accuracy.

The primary improvement is 10 net answers / 41.67 percentage points on this
specific development set, with two previously correct answers becoming wrong.

## Limitations

Only one seed and 24 previously exposed questions are evaluated. Numerical
variants and related question templates can remain semantically similar across
training and evaluation even with exact deduplication. Several settings change
from v0.2, so this run does not isolate each change's causal effect. A separately
authored blind benchmark and controlled ablations would be needed for stronger
claims; no further paid generation is authorized by this report.

## Artifact verification

The initial saved corpus checkpoint (Kaggle version 2, scriptVersionId
347579341) lists a 142,636-byte ZIP on its Output page. The reviewed corpus ZIP
was 159,930 bytes when verified in the live session. Browser download events
timed out, so no local archive download has been verified.

The saved adapter successfully reloaded against student revision
`0cb88a4f764b7a12671c53f0838cd831a0843b95`. All **392 adapter tensors** were
finite, and a fresh greedy response on `eval_01` exactly matched the stored
post-training raw response (`D`). This is a reload smoke test, not a repeat of
the entire evaluation.

The final ZIP passed CRC verification and all **124 manifest-listed files**
matched their sizes and SHA-256 hashes in the Kaggle runtime:

- File: `3beethoven_stats_flight_v0_3.zip`
- Bytes: **92,725,238** (about 88.43 MiB)
- SHA-256: `a6495b5f32f8e9f0d2adeb76582da0475988da3b8ceeed7d85f5257e3363a5b8`
- Contents: adapter/tokenizer, original and reviewed corpora, teacher caches
  without authorization headers, persistent API ledger, benchmark, splits,
  raw baseline/post responses, training logs, source snapshots, environment,
  summary, and verification record. Intermediate training checkpoints are
  excluded from this ZIP.

Final Kaggle **version 4**, scriptVersionId **347581888**, was saved using Quick
Save with output enabled. Its [Output page](https://www.kaggle.com/code/trinashih/3beethoven-v0-2/output)
was checked after the save and lists `3beethoven_stats_flight_v0_3.zip` at
**92.73 MB**, consistent with the verified archive size. The notebook retains
its historical v0.2 URL/title; the saved experiment and files are v0.3.

The final ZIP has not been successfully downloaded into the local workspace,
and no claim of a local, Library, or GitHub binary backup is made. The user
approved keeping the full archive on Kaggle for a later manual download.
Code and this report are committed to GitHub. Do not rerun paid generation to
recover the archive: download the saved output instead.

See [the protocol](STATS_V0_3_PROTOCOL.md) for masking, evaluation, budget,
checkpoint-selection, and persistence details.
