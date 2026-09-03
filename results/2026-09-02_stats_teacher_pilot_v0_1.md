# Statistics Teacher Pilot v0.1 — 2026-09-02

## Purpose
Validate whether Llama 3.3 70B can reliably generate synthetic training examples for the two statistical reasoning areas where the teacher/student gap is largest:

- inference / hypothesis testing
- distributions / expectation

This pilot follows the earlier teacher diagnostic, where Llama 3.3 70B scored 100% on both categories while Llama 3.2 3B scored 50% and 62.5%, respectively.

## Pilot configuration
- Teacher: `meta-llama/llama-3.3-70b-instruct`
- Provider: OpenRouter API
- Concepts: 24 fixed curriculum concepts
- One generated scenario per concept
- No free-form curriculum selection by the teacher
- Output: question, answer, explanation, common mistake

## Result
- Accepted: **24 / 24**
- Approximate API calls: **25**
- Acceptance rate: **100%**

This was dramatically more stable than the earlier classical-music teacher-data experiments. Constraining the teacher to a fixed concept list prevented topic collapse and duplicate generation.

## Initial manual review
The first six generated examples were materially cleaner than the classical-music pilot and correctly covered:

1. p-value interpretation
2. Type I error
3. Type II error
4. statistical power
5. alpha / power trade-off
6. confidence-interval width

No obvious core-concept error was found in these first six samples.

### Precision notes
- The p-value example is framed as a one-sided test and is acceptable, but future prompts should continue to specify test direction explicitly.
- The confidence-interval-width explanation assumes the usual standard-error relationship and a fixed estimator/model; this is acceptable for the intended introductory/intermediate curriculum but should not be overgeneralized.

## Decision
Proceed with statistics as the primary distillation candidate, focused on:

### Primary target
- inference / hypothesis testing

### Secondary target
- distributions / expectation

Do **not** use Bayes/probability as a primary teacher domain in this experiment because the 70B teacher itself scored only 50% there.

Do **not** prioritize regression/causality because the 3B student already scored 100% on the diagnostic, leaving no measurable distillation headroom.

## Experimental lesson
A large teacher model does not automatically make a good curriculum generator. The successful configuration was:

`fixed human-designed curriculum -> 70B scenario/explanation generation -> validation -> student training`

rather than:

`70B freely chooses what to teach -> student training`

The 24 examples in this pilot are still a pilot set. They should not be treated as the final training corpus until the larger generator includes validation and deduplication safeguards.
