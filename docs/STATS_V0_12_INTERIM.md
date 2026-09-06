# v0.12 procedural arithmetic experiment — preserved training checkpoint

Status as of 2026-09-06 02:27 PDT: training is complete and the selected weights are preserved in Kaggle version 25. The three-model evaluation was still running when the remote browser control environment restarted and lost its authenticated Kaggle session. This is an interim recovery record, not a final result report.

## Preserved training result

- Source adapter: v0.10, SHA-256 `14812770a7e612ab984e4ffad54bf514a3e00425655aa5adf732b975502f96f9`.
- Curriculum: 560 new procedural examples plus 516 unchanged v0.10 rehearsal examples; 1,076 training sequences total.
- Schedule: two epochs, 270 optimizer steps, learning rate `2e-5`, seed 1212.
- Selected checkpoint: step 135, chosen only by validation loss.
- Best validation loss: `0.15432609617710114`.
- Final reported training loss: `0.17937856162035906`.
- The runner verified the provisional archive at `/kaggle/working/3beethoven_stats_v0_12.zip`; displayed size was 92,802,432 bytes.
- Kaggle Quick Save version 25 was confirmed **Successful** with output saving enabled while evaluation continued.
- No teacher API calls were made; the procedural targets were generated and checked deterministically.

The provisional ZIP and selected adapter have not yet been independently downloaded or hashed. Version 25 is therefore a recoverable Kaggle weight checkpoint, not yet an independently verified binary backup.

## Evaluation observed before browser-control loss

The frozen runner compares v0.10, v0.11 and v0.12 on 80 procedural arithmetic questions, 48 new statistical-transfer questions and 240 old multiple-choice responses per model (1,104 responses total).

v0.10 completed before the interruption:

| Evaluation | Correct | Invalid | Token limit |
|---|---:|---:|---:|
| Procedural arithmetic | 7/80 | 27 | 5 |
| Statistical transfer | 3/48 | 17 | 18 |
| Old multiple choice | 130/240 | 0 | — |

The v0.10 arithmetic-category scores were GCD 2/20, multiplication 1/20, powers 4/20 and fraction reduction 0/20. v0.11 completed the 80-question procedural pass, but its scored summary had not yet printed when browser access was lost. Do not infer a v0.11 or v0.12 score from progress counters.

## Resume without retraining

Restore `trinashih/3beethoven-v0-2/versions/25` into a fresh Kaggle session. Preserve and hash-check its `3beethoven_stats_v0_12` directory and ZIP before doing anything else. The runner is resumable: `scripts/run_stats_v0_12.py` validates protocols and adapter provenance, reads existing response checkpoints, skips completed responses and continues missing evaluation rows. It must not repeat training when `training_complete.json`, the selected `adapter/` and matching protocol files are present.

After evaluation completes, run `scripts/verify_stats_v0_12.py`, independently inspect numeric answer formats, create the final report and results JSON, Quick Save a new final Kaggle version, download and verify the ZIP and adapter hashes, and only then stop the GPU.
