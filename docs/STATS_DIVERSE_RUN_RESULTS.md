# Training and Development Results — Historical Selection Record

The subsequent final comparison scored **45/180 for the original 3B and 170/180 for step180**. See the [final report](STATS_DIVERSE_FINAL_BLIND_RESULTS.md). The results below describe the earlier v15-based selection process, which remains `no_checkpoint_passed`.

## Training

- Parent v15 adapter SHA-256: `9369d52de4a886df9da0c872cd41bd4e01af0a38bf02ad724b5951c1a6b9f5d3`.
- 720 new examples plus 720 materialized historical training examples, alternating 1:1.
- 180 optimizer updates; last recorded loss approximately 0.20015.
- Final adapter SHA-256: `ff89a22f097e4db457dab287042ae36520ec6b9b36f26e5ed11fe42337c04f23`.

## Development scores

| Model | New development /180 | New paired losses | Unresolved pending | Legacy development /72 |
|---|---:|---:|---:|---:|
| v15 | 91 (plus 5 pending) | — | 5 | 51 |
| step60 | 139 | 2 | 2 | 69 |
| step120 | 167 | 0 | 1 | 72 |
| step180 | 172 automatic; 173 provisional | 2 | 0 after one provisional credit | 72 |

Every checkpoint had non-decreasing category totals relative to v15 on both development suites. All student legacy-development evaluations had zero paired losses and zero pending answers.

Step60 losses: `diverse_development_moment_variance_006`, `diverse_development_moment_second_007`. Pending: `diverse_development_process_scaled_002`, `diverse_development_moment_second_007`.

Step120 had no paired loss but retained pending question `diverse_development_process_scaled_002`.

Step180 losses: `diverse_development_moment_variance_006`, `diverse_development_binomial_009`. The owner provisionally credited `diverse_development_process_scaled_001`, raw `Expression: ((12/19)**2)*(10**2)`, exact value `14400/361`, raw SHA-256 `327520af57a1a3eb0098b0cd9ff40592aa9cec561976a44427bf0ba05df3fb37`. This post-output credit did not replace the automatic pending result or satisfy the frozen protocol retroactively.

## Execution deviations and stopping point

The v15 baseline was run after training, step180 was inspected before the prescribed 60→120→180 selection sequence, and the provisional credit was decided after viewing the output. No checkpoint passed every frozen gate. Legacy final was consequently not run for a selected student. At that historical stopping point, the final blind evaluation had not run; it was later authorized separately by the owner.

[Kaggle training archive, Version 62](https://www.kaggle.com/code/trinashih/3beethoven-v0-2?scriptVersionId=348379527) · [Original record with all receipts](STATS_DIVERSE_RUN_RESULTS.zh-TW.md) · [Historical JSON](STATS_DIVERSE_RUN_RESULTS.json)

[Preserved Traditional Chinese version](STATS_DIVERSE_RUN_RESULTS.zh-TW.md)
