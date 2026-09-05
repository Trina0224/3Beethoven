# Option-rotation diagnostic v1

This is the next diagnostic checkpoint, authorized after the 60-question
holdout showed strong answer-position effects. It uses the existing model and
questions only: **no teacher API requests, no training, no revised questions**.

Freeze this protocol before evaluating rotations. For each of the same 60
questions, rotate its four options left by 0, 1, 2 and 3 positions. Map the gold
letter accordingly; the answer content and question text never change. Each
question's correct option therefore occupies every letter exactly once.
These are four cyclic arrangements, not all 24 possible option permutations.

Run both vanilla and distilled 3B on all four arrangements (240 answers per
model), including a fresh original-order run. Use the unchanged primary
letter-only prompt, greedy generation, 16-token limit, strict parser, pinned
base revision and verified adapter hash. Resume only cached results under an
identical frozen protocol. Compare the fresh original-order raw answers with
the saved previous run and disclose mismatches.

Primary metrics: accuracy averaged across all rotations; number of questions
correct in all four arrangements; number selecting the same semantic option
across all arrangements. Also report constant-letter behavior, per-letter and
per-concept scores, invalid outputs and paired changes in all-four-correct
status. An always-A model scores exactly 25% on this diagnostic and has zero
semantic consistency; unit tests enforce this distinction.

The independent unit remains the question: **60 clusters**, not 240 independent
questions. This is a post-hoc diagnostic on an already exposed internally
authored holdout, not a new blind benchmark. No prompt tuning, key correction,
or retraining follows from these outcomes within this checkpoint.

Recover model and prior outputs by attaching the saved Kaggle version-5 output
as notebook input. Verify its adapter hash and benchmark hash before loading.
Preserve results in a new directory/ZIP, publish all raw answers and a report
to GitHub, and save a Kaggle version with outputs enabled. The original model
and holdout archives remain available in their earlier saved versions.
