# Multiview 歷史實驗：最終採用模型已更新為 V58

**已完成：最後採用學生 `final_clear`（Kaggle V58），直接從原 v15 接續訓練。** 同一份最終保留測試，v15 **107/144** → final_clear **144/144**；18 類全部不退步，9 類改善、9 類維持滿分。改正 37 題，保留原本正確 107 題，退步 0 題，待判 0 題。

目前結論與模型請見 [最終報告](STATS_FINAL_CLEAR_RESULTS.md) 與 [恢復指南](KAGGLE_RECOVERY.md)。V57 的局部成功不足以滿足整體優於 v15 的要求，先前「最後成功學生」定位已撤回。下方保留原始實驗數字及當時敘述供追溯，並非目前模型選擇或下一輪指令。

<details>
<summary>歷史紀錄</summary>

# Multiview pilot — completed

2026-09-07 17:52 PDT (America/Los_Angeles)

The explicitly authorized two-seed experiment is complete. All130 planned updates and1152 selection/probe generations are finished. Actual1032 microbatches and all archived file hashes verified; no pending semantic judgments. GPU stopped and UI confirmed.

New wording performance improved on the small crossed probe: known-number/unseen wording5/12→11/12 for both seeds; new-number/unseen wording4/12→7/12 and6/12. None of the eight checkpoints passed all unchanged original gates, so v15 remains preferred. Both step48 candidates preserve old/chain gates but still fail neither-event coverage; all weights remain available for research.

[Report](STATS_MULTIVIEW_REPORT.md) · [Full results](STATS_MULTIVIEW_RESULTS.json) · [Verified download receipt](STATS_MULTIVIEW_DOWNLOAD.json) · [All weight hashes](STATS_MULTIVIEW_WEIGHT_MANIFEST.json)

Saved Kaggle Version55, scriptVersionId348091555: [portable archive and full output](https://www.kaggle.com/code/trinashih/3beethoven-v0-2/output?scriptVersionId=348091555&select=3beethoven_stats_multiview_download.zip). Portable archive includes original v15 plus two final diagnostic students;279618858bytes,SHA25645646ed868207cf16723a105455d24dd8d95348a032a621fa43184657f123144. ZIP CRC and460 file hashes verified after download. Full output preserves all eight checkpoint adapters and optimizer states.

Do not rerun this experiment. Read the report's problem analysis and freeze a separate next protocol before more generation/training. The prior approval blocker was resolved by the user's explicit「允許」; no renewed permission is needed for already authorized actions.

## Historical preparation and resolved blocker

The following text records the pre-execution state; it is superseded by the completed result above.

> Execution update: user explicitly replied「允許」to the exact data-use/training/evaluation/save scope. The approval blocker is resolved. Original v15 weights and Version54 teacher/review data were restored and hash-verified; corpus regeneration and both contract tests passed in Kaggle. Prepared baseline and the two-seed driver have started. Prior blocked state below is retained as history. Kaggle SDK switches the mounted version for this same notebook when fetching Version42/54; use returned paths and copy each required artifact into a stable working input directory before switching versions.

# Multiview pilot execution checkpoint

Updated: 2026-09-07 16:39 PDT (America/Los_Angeles).

**Preparation complete; new training has NOT started. No model improvement result is claimed.**

## Frozen and verified

- 516 unique training prompts from 259 existing verified teacher rows: original and meaning-preserving alternate wording, minus two duplicate prompts. Every target remains byte-identical to its verified teacher source. Zero new teacher calls.
- Corpus train SHA256: 4dfe00b073e6ef805c7e83fbd18240e75c8a250749eaf5959911e41756a12a39.
- Probe SHA256: 64b870cd43e11dafa720f9243f99a2c1277f4c873908e92ddc971793fbbf1fd7.
- Two seeds, 2027/31415; original v15 parent; LR5e-6; 516 microbatches / 65 updates per seed; one uninterrupted pass. Checkpoints12/24/48/65. Original validation gates unchanged.
- 48-task numbers-by-wording probe is withheld from selection; known-number cells are fitting diagnostics. New task identities excluded against historical frozen splits, all consolidation candidates, selection and holdout.
- Target/oracle validation passed during generation. Two corpus contract tests passed, including exact same wording along the numeric axis; Python compilation passed.
- Decision frozen before generation/training: commit f21c94ba1f2587e0351e224ac83f3b8ecab6a4b9. Runner later changed only the provenance label for mounted review records.
- This changes wording and exposure together. No causal claim isolating augmentation and no promotion claim from this filtered pilot.

## Execution blocker

Kaggle session started and pinned package installation completed. The newly started draft automatically mounted Version54, whereas the prior session used Version53. Version54 has the verified teacher rows and review records but not the original v15 parent adapter. The intended setup restores the hash-verified original parent from this same notebook's Version42, and reads the already mounted Version54 teacher/review files.

Automatic approval review rejected copying these existing private weights, teacher data and review records into the executing Kaggle workspace. Read-only checks verified the destination is the SAME notebook URL as the source notebook, and prior personal context recorded permission to restore Kaggle weights and continue training/evaluation. Automatic review still rejected the action and explicitly requires current user approval; it did not accept the recovered prior authorization as sufficient.

No baseline generation, new student optimizer updates, or probe generation occurred. The partial Kaggle setup must NOT be treated as complete: an attempted chunked review-file transfer was interrupted by approval rejection, so that partial file must be rebuilt from mounted evidence and validated before any execution. The full valid local/GitHub corpus and review registry remain available. GPU stop requested after the final rejection; verify stopped UI before ending the turn.

## Exact approval needed

Permission to read/restore existing original-v15 model weights, verified teacher data and semantic review records from earlier versions of `trinashih/3beethoven-v0-2` into the executing session of that same Kaggle notebook, then run and save this frozen two-seed experiment. This is not permission to send the data to another person or unrelated workspace.

## Resume after approval

1. Start the same notebook; restore original v15 from Version42 and assert adapter SHA2569369d52de4a886df9da0c872cd41bd4e01af0a38bf02ad724b5951c1a6b9f5d3.
2. Rebuild a complete source tree and review registry from mounted Version54; overlay the committed multiview runner, corpus generator, decision, and current baseline helper. Recreate corpus and validate all digests/tests. Do not trust partial files from this stopped draft.
3. Run prepared v15 baseline, then both fixed one-pass student runs; verify actual 516 input/label/order hashes per seed. Resolve pending judgments after training, then freeze earliest full-gate pass or final diagnostic per seed.
4. Only then run the crossed probe on v15 and two fixed candidates. Preserve raw outputs, all failures and automatic scores; no answer-only semantic credit.
5. Save full results, selected/final diagnostic weights and portable archive to a completed Kaggle version; verify download and stop GPU. Keep v15 preferred unless the established promotion evidence exists.

</details>
