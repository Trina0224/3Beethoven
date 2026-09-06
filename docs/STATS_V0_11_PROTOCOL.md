# v0.11 frozen arithmetic follow-up

Authorized continuation of the statistics pilot: investigate high memorization risk without assuming the student learned nothing. This is exact algorithmic arithmetic supervision mixed with existing Llama teacher rehearsal, not a new pure teacher-distillation comparison.

## Frozen data

Seed 1111. Five operations: fraction multiplication, fraction addition, integer powers, fraction powers, fraction reduction. 400 new training questions (80 each), 40 validation (8 each), 60 canonical test (12 each). A further 60 equivalent-representation questions belong to those SAME test groups, not independent questions. Fraction inputs use random values with denominators 7–60, powers 2–5; integer bases 2–60. Canonical rational identities prevent scale/swapping leakage. Final answer values are disjoint across new train/validation/test/transfer and exclude v0.9/v0.10 benchmark answer values. Thus an equivalent scale version never moves between splits. These controls cannot exclude base-model pretraining memorization.

Forty-eight new statistics transfer questions (24 binomial,24 exactly-one detection) use denominator 60/80, exclude rational parameter identities from every v0.9/v0.10 split, and exclude new arithmetic answer values. They are not new task families. Training contains NO v0.11 transfer questions, answers or substep-derived examples. No exposed v0.10 test questions are added to training.

Rehearsal: unchanged 516 v0.10 training and 64 validation sequences, containing earlier audited Llama teacher responses and MC targets. Totals 916 training /104 validation sequences. Exact arithmetic targets are generated with rational arithmetic; every equality in every target is verified. Mixed examples are shuffled by Trainer. No new teacher calls or API charges.

## Frozen training

Start v0.10 adapter SHA256 14812770a7e612ab984e4ffad54bf514a3e00425655aa5adf732b975502f96f9, restore Kaggle version18. Same Meta Llama3.2-3B-Instruct base revision 0cb88a4f764b7a12671c53f0838cd831a0843b95 and LoRA architecture. Fresh optimizer, 2 epochs, LR3e-5, seed1111, batch1 accumulation8, warmup5 cosine, 4-bit NF4 double quantization float16, paged AdamW8bit. No sequence truncation permitted; cap768. Best checkpoint chosen solely by validation loss. Train first, save weights, then compare v0.10 and reloaded v0.11 with identical inference settings. Test results never select checkpoints.

## Scoring and goals

Numeric 256 tokens, greedy, identical Formula/Calculation/Answer prompt. Exact rational equality, no floating tolerance. Reduction questions additionally require lowest-terms output; no credit for unsimplified equal fractions. Strict score requires one Answer line. Conservative automatic format review permits an explicit final scalar after a terminal equals sign or standalone scalar when no Answer line exists; it never computes unfinished formulas or overrides a wrong final answer. Audit raw outputs independently before final conclusions. V0.10/v0.11 are rescored with these same rules; historical rounded-tolerance scores are not substituted.

Primary: canonical arithmetic reviewed score improves >=12/60 AND reaches >=30/60. Secondary transfer improves >=6/48. Retention old60 questions x4 rotations no worse than v0.10 minus4/240. Report equivalent-representation accuracy and paired both-correct separately; do not double the apparent independent sample count. Full protocol success requires all frozen goals. Failure remains a valid result.

Limits: one seed, small test, hybrid curriculum, no pure SFT-versus-distillation or calculator intervention causal claim. No calculator at inference. Final-answer uniqueness reduces one leakage route but cannot prove abstract algorithm learning. Exact-answer questions are intentionally harder than two-decimal matching. No post-test retuning in this run.

## Artifacts

STATS_V0_11_FROZEN_QUESTIONS.json, curriculum/scoring tests, run script, all raw outputs, environment and train logs, selected adapter, hash manifest and ZIP. Save Kaggle outputs and independently back up exact adapter bytes. GitHub stores text/code/data/report, not weight binaries.

