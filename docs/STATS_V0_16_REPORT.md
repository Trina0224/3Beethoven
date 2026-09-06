# v0.16 completed: stronger aided formulation, weaker unaided total

The student was trained through all three planned stages. It is not promoted as a general replacement for v15: on the same new 96-question unaided test, reviewed formulation accuracy falls from **34/96 to 29/96**. With task-specific symbolic rules, the two target families improve from **11/24 to 22/24 (91.7%)**. This supports a rule-assisted specialist candidate, not independent mastery.

## Same-run results

| Condition | v15 automatic | v15 reviewed | v16 automatic | v16 reviewed |
|---|---:|---:|---:|---:|
| No formula reminder, 96 new questions | 33 | **34** | 26 | **29** |
| Formula reminders, 24 target questions | 9 | **11** | 15 | **22** |
| Old 240 option rotations | 127 | 127 | 126 | 126 |

| Unaided family, 12 questions each | v15 | v16 |
|---|---:|---:|
| Second moment | 0 | **3** |
| Affine Poisson variance | 1 | **4** |
| Poisson time conversion | 7 | **9** |
| Conditional total waiting time | 5 | **0** |
| Binomial probability | 1 | 0 |
| Exactly one detection | 8 | 5 |
| At least one detection | 12 | 8 |
| Interval upper endpoint | 0 | 0 |

Unaided: 10 newly correct, 15 newly wrong, 19 correct in both. On the six non-target families, 12 previously correct answers were lost; the retention target of at most three losses was missed. Both weak-family targets of at least 8/12 unaided were also missed. The low validation loss did not predict broad test transfer.

With rules, second moments improve **7/12 to 10/12** and affine Poisson variance **4/12 to 12/12**. Paired aided outcomes: 13 newly correct, two newly wrong, nine correct in both. Aided and unaided endpoints remain separate.

## What was trained

Start from saved v15. 48 fresh training stories contrast scale identification, E[Y], Var(Y), and E[Y²]. Initially 150/192 training and 21/36 validation responses passed the frozen automatic structure checks. A separate teacher-only review recognized correct evaluated scale squares, such as 9*variance for scale 3. Final accepted targets: **188 train, 36 validation**, plus **48 historical training replay examples**. Four truly incorrect targets were excluded. Teacher responses were never repaired with numerical gold. All original rejects and scores remain saved.

Teacher usage: **361 calls, US$0.010648875**, all response costs reported. No extra calls were needed after reviewing equivalents. The teacher received focused symbolic rules, not numeric answers or test questions.

Each stage contains 236 sequences. Full symbolic reminders: one epoch/30 steps; quantity-only cues: one epoch/30 steps; no reminders: three epochs/90 steps. Total **150 optimizer steps**, learning rate 2e-5, seed 1616, effective batch eight. Final selected checkpoint: `none/checkpoints/checkpoint-90`.

Unaided validation loss: 0.194882 after full support, 0.076093 after cues, and 0.009221 at the selected final checkpoint. Checkpoint selection used validation only; test outputs were generated after selection. This bundles contrasts, fading support and more compute; no single-factor causal claim.

## Review and interpretation

All 240 formulation responses were read, including wrong and pending outputs. The frozen automatic grades are unchanged. Supplemental review credits three v16 unaided and seven aided expressions that correctly evaluate the scale square. It credits one v15 binomial expression written with “choose” notation and two v15 aided moment setups explicitly present before arithmetic-only continuation. One of the latter later adds incorrectly by 100; it receives formulation credit, not final-answer credit, consistent with the project endpoint. Exact decisions and raw hashes are in `STATS_V0_16_SEMANTIC_REVIEW.json`.

The remaining aided v16 moment failures put the squared scale into the mean term as well: e.g. `16*324+(16*32+11)**2` instead of retaining multiplier 4 inside the mean. Unaided outputs also confuse mean versus variance or multiply the squared mean by the squared scale twice. This is more than an arithmetic problem.

The output instruction differs from v15's earlier 64-question test and does not explicitly list `comb(n,r)` as an allowed function. The question parameters also differ. Both checkpoints use identical prompts within this run, so their paired comparison is valid, but the earlier 44/64 must not be interpreted as directly falling to 34/96. Binomial/interval weakness in both checkpoints may involve prompt sensitivity; this run does not isolate that cause. The new test uses known wording families with fresh identities, not unseen concepts. Training still contains no test identities.

## Decision

Retain **v15 as the general unaided candidate** and preserve **v16 as a rule-assisted specialist candidate**. Do not continue blind epoch increases on this mixed curriculum. A practical integration would supply task-specific symbolic rules and let the student bind numbers, with arithmetic delegated to the calculator. The measured 22/24 belongs only to two supplied-rule task families; an end-to-end router/skill system has not been tested or deployed. General rule selection and broad independent transfer remain open.

## Verification and backup

All **720 answers**, **392 finite adapter tensors**, **737 manifest files** and ZIP integrity verified. Adapter SHA-256: `117a009f72ebafe6e6baefef62a6b81e7bbcefbc902f7eb3d93f5e73f48d46d0`. Final ZIP: **371,440,108 bytes**, SHA-256 `494fd49d497ab1569578eb4983d22d90bd2721495bb1d683f46b3a26c385067f`. The ZIP includes the intermediate adapters as well as the final adapter.

Kaggle Version 31 successfully preserves the completed training checkpoint. **Version 32 is Successful** and preserves the complete output with Always Save Output enabled; recover the final adapter from `3beethoven_stats_v0_16/adapter/adapter_model.safetensors`. Binary weights remain on Kaggle; GitHub preserves code, original teacher responses, results, review and recovery instructions. No independent fresh-session restore of v16 has been claimed.

The browser progress connection ran out of memory during training; reconnecting confirmed Kaggle was still running. No training was restarted because of that browser failure.
