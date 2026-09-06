# v0.15: decomposed teaching for weak formulations

## Decision recorded before generation and training

The unassisted 70B teacher previously omitted the squared-mean contribution in
three second-moment responses. General formula reminders fixed the moment items
in a new diagnostic, but that diagnostic was not the v0.14 student curriculum.
The v0.14 student scored 0/8 on second moments, affine Poisson variance and total
conditional waiting time. We will explicitly teach intermediate quantities and
increase practice, without declaring failed items out of scope or deleting them.

Teacher reliability and student learning are separate metrics. Preserve all
teacher attempts, reject wrong formulas, and report reminders supplied to the
teacher. Correct selected training answers do not guarantee a student can learn
the concept from the amount and form of supervision supplied.

## Frozen curriculum

- 168 new teacher training candidates: 12 per original family, 12 additional per
  weak family (moment, affine Poisson variance, Poisson time, conditional wait),
  and 24 intermediate moment tasks (mean and variance for 12 matching stories).
- 32 new validation candidates, four per original family.
- 64 new test questions, eight per family, never sent for teacher generation.
- Exclude problem identities from v0.13/v0.14 and both prior teacher diagnostics.
  Split grouped moment stories together. Train/validation/test are disjoint in
  problem identity; numerical answers may coincide by chance.
- Add 32 verified v0.14 TRAINING examples (four per family) for replay.
- Teacher gets general symbolic formula reminders, never numeric gold answers.
  At most two responses per candidate, 400-call cap; select first verified answer.
  Teacher AST normalization never fills missing values from the reference.
- Student gets word problems and one-line expression targets; no formula reminders
  are included in the student's test prompt.
- This is a curriculum package; the run does not isolate which individual change
  caused an effect. Do not describe it as a controlled decomposition-only ablation.

## Training and evaluation

Start v0.14 adapter SHA-256
`c7def77757fefaaf41db6938500159795a47503dac54d72d79113de47a3239a5`.
Three epochs, learning rate 2e-5, effective batch eight, seed 1515. Select checkpoint
by validation loss only. Compare vanilla, v0.14 and v0.15 on the SAME new test and
the 240 old multiple-choice rotations. Report raw and reviewed counts separately.

The v2 scorer adds only equivalences identified in v0.14, before v0.15 responses:
Comb capitalization, correct binomial complements/exponents, alternate event sums,
and Poisson rate-times-duration multiplication/division association. Wrong values
remain wrong; unknown equivalences stay pending. All previous raw scores remain.

The frozen question digest is
`d7df943fa318ffbaeffabd12834a8f345c54f71c48f8c8cc1cbef8c78ba55f1f`.
Success evidence is positive paired test improvement and gains in weak families,
reported alongside retention. Do not tune on test outcomes or require another
teacher aggregate gate after filtering every training target individually.

## Preparation revision (before student training or test evaluation)

Initial acceptance was 133/168 training and 23/32 validation, below the required
validation minimum. Genuine errors included squared Poisson rates and interval
endpoints anchored at the lower bound. Initial records are preserved unchanged.
A separate focused-teaching pass makes at most ONE new response for each of the
44 rejected candidates, using only that family's symbolic rule. No numeric gold
answers are sent, scoring and test questions remain frozen, and initial pass rates
are not replaced by supplemented rates. This is an explicit preparation revision,
not an undeclared third retry under the original two-attempt protocol.
