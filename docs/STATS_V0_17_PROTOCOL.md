# v0.17: exact adapter mixtures and conservative rehearsal

Authorized by the user after reviewing v15/v16 tradeoffs. Frozen before any v17 model outputs.

- Start training from hash-verified v15. Compare v15, v16, exact delta mixtures with v16 weights 0.25/0.50/0.75, and training steps 8/16/32.
- Merge uses concatenated low-rank factors, not separate A/B averaging. Rank and alpha both double to retain original scaling. Adapter memory increases; this is not a free same-size merge.
- Reuse only accepted Llama TRAIN responses: 24 per each of eight families (192), plus 64 v16 same-story contrast responses. No new teacher calls. No validation or test answers enter training.
- Training: 256 sequences, effective batch 8, 32 steps, lr 5e-6, cosine schedule, two warmup steps, seed 1717. Checkpoints at 8/16/32 are the preregistered candidates (step24 is saved only for recovery).
- Fresh balanced validation: 48 questions (six per family), never used for gradient updates. Select by frozen formulation correct count; ties favor non-target-family correct count, then minimum family count, then earlier candidate in declared order.
- Choose best interior mixture and best trained checkpoint on validation; also select overall candidate including v15/v16. Seal choices before final test.
- New test: 96 questions, 12 per family, all identities disjoint from v13–v16 and prior teacher probes. Same known wording families, not general reasoning or unseen-template evaluation.
- Test v15, v16, selected mixture, selected retrained checkpoint. Old 240 rotations run for v15 and the overall validation winner.
- Explicit expression grammar includes comb(n,r), identical for all candidates. This differs from v16's prompt; do not compare absolute scores across runs.
- Frozen grading carries prior equivalences and evaluated scale-square rules forward. Exact numeric coincidence alone never earns formulation credit. Preserve pending outputs and raw grades; supplemental semantic review cannot select a different model after test.
- Promotion goal: at least six additional correct new-test formulations over v15; no more than three v15-correct losses across six non-target families; old MC within four of v15. Report each gate and per-family tradeoffs.
- Preserve all source weights, selection data, raw responses, training logs, finite-tensor checks, ZIP manifest and hashes. Preserve completed Kaggle output before stopping GPU.

Frozen data SHA256: `6b686160203e15ca9cc6be270ec2fe98efeccfcf961be560505af3d2eb477908`.
