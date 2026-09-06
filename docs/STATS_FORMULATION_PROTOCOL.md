# Formulation audit and calculator-assisted evaluation

This is a new, exploratory endpoint proposed after earlier numerical results
were observed. It does not replace any frozen v0.9–v0.12 endpoint or declare
that those original success thresholds passed.

## Separate measurements

1. Formulation: correct event, distribution, assumptions and symbolic formula.
2. Substitution: correct assignment of the question's quantities to the formula.
3. Execution validity: the model's expression fits the calculator interface.
4. Assisted final accuracy: executing that expression gives the reference value.
5. Unaided final accuracy: preserve the original score without modification.

Report formulation and substitution separately. A correct final number alone
does not establish correct formulation. A bare formula without needed parameter
bindings is not a complete executable solution. Equivalent formulas are valid;
ambiguous or truncated reasoning is marked unassessable, not silently credited.
Report correct/incorrect/unassessable counts over the full sample, alongside
correct counts among assessable responses. Never omit unassessable cases from
the headline denominator.

For retrospective review, preserve exact evidence spans and separate verdicts
for each response. Review all eligible responses, with model labels hidden where
practical. Record reviewer identity/process and disputed cases. Do not use the
correct answer to repair a student's formula or supply missing parameters.
Retrospective execution of extracted expressions is an audit condition, not
evidence that the model independently learned tool calling.

## Prospective interface

Provide the same interface and token allowance to each compared model. Request
one expression using integer literals, parentheses, +, -, *, /, **, comb(n,k)
and gcd(a,b). Use fractions rather than decimal literals. The executor is
`scripts/exact_calculator.py`. It has no question, reference answer, file access,
network access, arbitrary Python evaluation, or automatic formula repair.

This executable module is the calculation backend for a future skill wrapper;
it is not yet connected to the student or installed as a model skill.

Freeze the new prompts, parser and scoring before new inference. Give the
unmodified student and each trained student identical tool access: gains from
the calculator alone must not be attributed to distillation. Keep teacher-made
rehearsal and deterministic arithmetic targets distinct in provenance. Compare
matched questions; report this as new-instance evaluation, not unseen-skill
generalization. Existing exposed questions can be used for exploratory audit,
but a confirmatory claim requires a separately held-out evaluation.

## Current status

Calculator backend and unit tests implemented. No formulation scores or
calculator-assisted student scores have been measured yet. v0.12 final results
remain unverified following loss of authenticated Kaggle access; version 25 is
the previously confirmed selected-weight checkpoint. No new training or paid
teacher requests are required for this local implementation step.
