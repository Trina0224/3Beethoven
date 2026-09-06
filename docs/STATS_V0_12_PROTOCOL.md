# v0.12 procedural arithmetic experiment

This is the authorized alternative-method follow-up. It tests whether concise, explicit execution steps can turn correct statistical setup into correct final values. It does not assume that v0.11 merely memorized answers.

## Design frozen before training

v0.12 restarts from the v0.10 adapter, rather than continuing from v0.11, so it is an alternative treatment from the same source. It mixes 560 deterministic procedural examples with the unchanged 516-example audited Llama response-distillation rehearsal set. Validation mixes 64 new examples with the unchanged 64-example rehearsal validation set. No teacher API calls are made.

The new curriculum contains four equally sized arithmetic skills: place-value multiplication, repeated-power calculation, Euclidean GCD, and fraction reduction with an explicit GCD. It also contains balanced full pipelines for binomial exactly-r probability and exactly-one-detection probability. Targets expose powers, integer products, denominators, GCDs, reduction, and one final Answer line. They remain short enough to avoid teaching long looping prose.

New train/validation and test identities are disjoint. Final answers are disjoint across all new splits and exclude answer values in v0.9, v0.10, and v0.11 data. Statistical training/validation uses denominator 50 or 100; frozen transfer uses denominator 70 or 90 and excludes all earlier statistical parameter identities. This blocks exact prompt/parameter and final-answer reuse, but cannot prove absence of related base-model pretraining.

## Training and comparison

Source is v0.10 adapter SHA-256 `14812770a7e612ab984e4ffad54bf514a3e00425655aa5adf732b975502f96f9`. Base is Meta Llama 3.2 3B Instruct revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`. Train two epochs with seed 1212, LR 2e-5, batch 1 and accumulation 8, warmup 5, cosine schedule, 4-bit NF4 double quantization float16, and paged AdamW 8-bit. Select the checkpoint only by validation loss. No sequence truncation is allowed under the 768-token cap.

Compare identical greedy inference for v0.10, v0.11, and v0.12 on 80 new micro-arithmetic questions, 48 new statistical-transfer questions, and the old 60-question four-rotation benchmark. Numeric generation is capped at 192 tokens. Exact rational equality is required; a fractional final answer must be in lowest terms. One explicit `Answer:` line is required. Raw outputs receive independent review before conclusions.

Frozen goals compare v0.12 with its v0.10 starting model: micro arithmetic gains at least 16/80 and reaches 48/80; statistical transfer gains at least 8/48 and reaches 24/48; old-benchmark retention is no worse than minus 4/240. Full success requires every goal. v0.11 is a diagnostic comparison and does not change these thresholds.

Limitations include one seed, a small synthetic test, hybrid rehearsal, exact-output sensitivity, and no causal separation among formatting, curriculum structure, and arithmetic content. A failed result remains valid. Do not tune again on these test results within v0.12.
