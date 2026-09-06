# v0.14 student response distillation

## Fixed before student evaluation

- Start: v0.13 adapter, SHA-256 `7549a593f282e01b5b0b985dd20b5b2df0e52c390b0822ed637197327e2816ca`.
- 126 training / 31 validation examples from independent Llama 3.3 70B responses.
- Select the earliest verified response from at most two saved attempts. Targets
  expand only teacher-owned assignments and normalize decimal spelling. Never
  copy reference formulas or insert missing reference bindings into targets.
- One-line numerical expression output. The prompt does not include formula reminders.
- Two epochs, learning rate 2e-5, effective batch 8, seed 1414; select checkpoint
  using validation loss only. The test never selects the checkpoint.
- Compare vanilla 3B, v0.13 and v0.14 on the identical frozen 64-question wording/
  parameter test and 240 old multiple-choice rotations. Preserve every response.
- Score formulation using the frozen representation-tolerant grader. Pending
  semantic review is reported separately. Numeric equality alone is insufficient.
- Same eight statistics concepts; this is not a general reasoning benchmark.
- This experiment combines a procedural-SFT starting checkpoint with a new
  teacher-response training stage. It does not isolate distillation from scratch.

## Teacher diagnostics

The original independent diagnostic remains failed: automatic 41/48, semantic
review 43/48. General symbolic formula reminders were then frozen and evaluated
on a new 48-prompt set, disjoint in problem identity from that diagnostic and the
v0.14 student splits. The new diagnostic passed unchanged thresholds: **46/48**,
22/24 complete pairs. Cost: US$0.00269837. One response is incomplete (`n`), and
one incorrectly squares a Poisson process rate. No responses were retried.

This is evidence for a rule-scaffolded teacher, not a correction to the independent
diagnostic's score. The student corpus remains the earlier independently generated,
individually verified teacher answers; these new diagnostic examples are never
included in training. No teacher formulas are supplied during student evaluation.

## Execution

Pinned launch revision: `4288ab82109df631f92da3454f51ee5357d97ffe`.
Kaggle T4 x2 environment; the established QLoRA runner uses GPU 0 only.
Training, three-model evaluation, ZIP creation and tensor verification are chained
in one cell. Results and final backup identifiers will be added after verification.
