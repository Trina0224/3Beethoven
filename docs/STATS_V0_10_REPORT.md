# Statistics v0.10 final report

Status: completed; partial improvement, primary success criteria not met.

## Comparable evaluation

All models used the same frozen 48 new questions. MC evaluates four option rotations (192 responses, not 192 independent questions). Numeric uses 48 free responses.

| Model | New MC | Numeric strict | Numeric format-reviewed | Old MC |
|---|---:|---:|---:|---:|
| Base | 52/192 (27.08%) | 0/48 | 4/48 | Not run |
| v0.9 | 67/192 (34.90%) | 13/48 | 13/48 | 127/240 |
| v0.10 | 87/192 (45.31%) | 20/48 | 20/48 | 130/240 |

Numeric gain is seven questions (+53.85% relative, +14.58 percentage points), with eight newly correct and one newly wrong. This does not double v0.9. Frozen goals: gain at least eight FAILED; at least 24/48 FAILED; old MC no worse than v0.9 minus four PASSED. No further tuning on this exposed test set.

## Independent format and arithmetic review

All 144 numeric outputs were reviewed. Baseline and v0.9 UI review texts were matched to canonical exported raw text after whitespace/JSON escaping normalization. Baseline gains credit only for explicit final correct scalars in IDs type_i_010 and type_ii_025,036,044. No v0.9 or v0.10 format corrections. An unfinished correct expression is not a correct final answer, and a correct intermediate followed by a wrong final answer receives no credit. This review supersedes the pending-review marker preserved in raw RESULTS.json.

Exact rational evaluation of the first numerical expression after Calculation found 42/48 correct for v0.9 and 45/48 for v0.10. These are diagnostics, not answer-score replacements. V0.9 incorrect initial substitutions: type_i_005,019; type_ii_026,035,037,046. V0.10: type_ii_032,035,038.

V0.10 binomial initial expressions are correct in 24/24, yet final answers only 3/24 (v0.9 1/24). Detection-probability answers improve 12/24 to 17/24. Most remaining failures occur in powers, multiplication, denominators, and reduction. Example type_i_014 reaches 3430/32768 then wrongly reduces to 1705/16384; expected 1715/16384. Type_ii_026 reaches 514/1600 then wrongly reduces to 127/400. Two responses (type_i_009 and type_ii_046) repeat expressions until the fixed 256-token limit, without a final answer. The evidence supports an arithmetic execution bottleneck, not a claim that all conceptual errors are solved.

## Training and limitations

Continued the v0.9 LoRA with fresh optimizer, 130 steps, two epochs, learning rate 2e-5, effective batch eight; selected checkpoint 130 using validation loss 0.1792280674. Teacher generation and evaluation used 269 calls, reported cost $0.020955115. Teacher original-order MC is 29/48; this is not directly the four-rotation student aggregate.

Training combines new denominator-40 examples, staged solutions and rehearsal. It is not a format-only ablation. 75/112 teacher records used reference-conditioned generation; 37 of those were verbatim reference format repairs. Frozen test parameters are separate, but task families are trained; this does not establish general mathematical reasoning transfer. One seed and a small test set limit generalization. Option-position bias persists; only 2/48 questions are correct under all four rotations.

## Verification and recovery

Verified 1,200 student responses, 392 finite adapter tensors, and 461 manifest files. Selected adapter SHA-256: `14812770a7e612ab984e4ffad54bf514a3e00425655aa5adf732b975502f96f9`.

Final archive: `3beethoven_stats_v0_10.zip`, 93,104,858 bytes, SHA-256 `470e4013b2f11ef52e6bd60736f73a1121e66e0bfe757093a8d3fd4e9affc677`.

GitHub preserves complete raw results in STATS_V0_10_RESULTS.json and training data/code/protocol. Actual weight backup locations and final save verification are tracked in MODEL_BACKUP_STATUS.json. GitHub does not contain weight binaries.

Next recommended experiment: a frozen arithmetic curriculum focusing on exact fraction operations and intermediate-step verification, with separate unseen parameters. A calculator-assisted diagnostic can quantify the reasoning/execution gap, but must be reported separately from unaided student scores.
