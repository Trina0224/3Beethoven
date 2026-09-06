# v0.17 completed: conservative retraining ties v15; mixture gains one question

Both requested approaches were executed: exact v15/v16 adapter mixtures and a new balanced rehearsal run from v15. **Neither meets the preregistered promotion goal.** Keep v15 as the general candidate, preserve v16's complementary specialist capability, and retain the new checkpoints and mixtures as reproducible experimental artifacts.

## Same fresh 96-question test, without formula reminders

All 384 final-test expressions were read against the requested quantity and reference. The semantic review confirms these frozen automatic totals without additional test credits.

| Candidate | Correct formulations | Newly correct vs v15 | Newly wrong vs v15 | Old MC |
|---|---:|---:|---:|---:|
| v15 | 64/96 | — | — | 127/240 |
| v16 | 57/96 | 20 | 27 | Not rerun in this experiment |
| 75% v15 + 25% v16 delta mixture | 65/96 | 5 | 4 | Not rerun in this experiment |
| v17 retrained, selected step 8 | 64/96 | 1 | 1 | 127/240 |

The mixture gains four affine Poisson variance problems and one interval problem, but loses four conditional-wait problems. The retrained candidate fixes one Poisson-time conversion and loses one conditional-wait formulation. Its old-MC score is retained, but there is no net new-test gain.

| Family, 12 questions each | v15 | v16 | 25% mixture | v17 step 8 |
|---|---:|---:|---:|---:|
| Poisson time conversion | 9 | 5 | 9 | 10 |
| Affine Poisson variance | 1 | 12 | 5 | 1 |
| Second moment | 0 | 9 | 0 | 0 |
| Conditional total wait | 9 | 0 | 5 | 8 |
| Binomial probability | 12 | 12 | 12 | 12 |
| Exactly one detection | 12 | 7 | 12 | 12 |
| At least one detection | 12 | 12 | 12 | 12 |
| Interval upper endpoint | 9 | 0 | 10 | 9 |

**v16 can formulate many target-family problems unaided in this condition:** second moments 9/12 and affine Poisson variance 12/12. This updates the narrower earlier observation of rule-assisted strength. The failure is retaining those gains together with the other families in one general candidate, not evidence that no target skill transferred.

## Validation selection, sealed before test

| Candidate | Frozen automatic /48 | Supplemental semantic /48 |
|---|---:|---:|
| v15 | 33 | 34 |
| 25% v16 mixture | 33 | 33 |
| 50% v16 mixture | 29 | 30 |
| 75% v16 mixture | 31 | 32 |
| v16 | 28 | 29 |
| Train step 8 | 34 | 36 |
| Train step 16 | 34 | 36 |
| Train step 32 | 34 | 36 |

Selection used the frozen formulation score on 48 new balanced validation questions, followed by non-target correct count, minimum family count and declared candidate order. This selected step 8 overall and the 25% interior mixture. The full response-level semantic review adds ten validation credits: commuting the rate and seconds factors with the same division by 60, or evaluating an interval denominator 2*2 to 4. It leaves both choices unchanged. Original scores, responses and selection remain immutable; final test scores did not select the mixture or checkpoint.

## Training and exact mixture implementation

The run starts from hash-verified v15. It reuses 256 accepted Llama training responses: 24 per each of eight families (192), plus 64 same-story mean/variance/moment/scale contrasts from v16. No old validation or test responses become training targets. There are **zero new teacher calls** and zero new teacher API cost.

Training uses lr 5e-6, effective batch 8, seed 1717, two warmup steps, cosine scheduling and 32 optimizer steps (one epoch, 149.7 seconds). Steps 8, 16, 32 are the planned candidates; step 24 is saved for recovery only. Selection retains step 8 even though the full 32-step run completed. This is a bounded conservative setting, not a search over all learning rates or curricula.

Mixtures implement the exact low-rank update sum `(1-a)*B15*A15 + a*B16*A16` through factor concatenation. They do not independently average A/B factors, which would introduce cross terms. Rank and LoRA alpha both double to preserve scaling, so adapter size increases. Each layer's concatenated update is numerically checked against the direct weighted sum. All candidates share the pinned Llama base revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`.

## Decision and limits

The selected retraining passes the non-target retention limit (one loss, maximum three) and old-MC limit (no score loss, maximum four), but fails the required six-question fresh-test gain. The mixture also fails the gain goal and has four non-target losses, exceeding the maximum three; its old-MC gate was not measured because it was not the overall validation winner. Neither is promoted.

This evaluates fresh parameter identities in eight known wording families, not unseen concepts or general mathematical reasoning. The output prompt explicitly permits `comb(n,r)` and is identical for all candidates. It differs from the earlier v16 experiment, and the question parameters differ too: do not attribute cross-run score changes solely to training or solely to prompting. No formula reminders or numerical references appear in these student test prompts. All validation and final-test outputs fit within the 160-token cap.

The evidence supports complementary learned capabilities, but this experiment does not establish a successful single-model consolidation. Other merge methods, routing and training schedules were not tested here.

## Preservation and verification

The run preserves all 1,248 generated responses (384 validation, 384 final-test, 480 old-MC), source, accepted teacher targets, selection record, checkpoints and adapter mixtures. A separate review records decisions for all 768 formulation responses. The verifier checks question/prompt hashes, frozen scores, selection, finite adapter tensors, manifest hashes and ZIP CRC.

Kaggle Version 33 preserves the completed training checkpoints. Final complete-output version and archive hashes are tracked in `MODEL_BACKUP_STATUS.json`. Binary weights are saved on Kaggle; GitHub preserves code and text results. v15 and v16 were independently restored into this fresh session, hash-checked and used successfully. A fresh-session restore of the new v17 archive is not claimed.

The inherited browser draft initially failed with a sequence-number concurrency conflict. Reloading the editor resolved it while the underlying GPU process continued; no successful training was repeated.
