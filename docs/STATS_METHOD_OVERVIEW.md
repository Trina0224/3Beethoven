# Curriculum and Evaluation — English Overview

This is a reader's guide to the existing curriculum and historical frozen protocol, not a replacement protocol or a new training authorization. The full original contracts remain linked below.

## Bounded task

Emit a fully substituted numerical expression for a direct statistics question. Arithmetic is delegated to a calculator. Training targets must pass strict one-line formatting checks. Student mathematical credit is assessed separately from formatting: a safely extracted, executable, mathematically correct expression can receive credit even when the whole response fails strict format.

## Coverage

| Family | Categories | Same-story contrasts |
|---|---|---|
| Independent events | both, neither, exactly_one, at_least_one, same | Five event questions for the same probabilities |
| Affine moments | moment_mean, moment_variance, moment_second | Mean, variance, second moment for Y=aX+b |
| Poisson count | poisson_variance, poisson_scaled, poisson_second | Three quantities for the same count |
| Poisson process | process_variance, process_scaled, process_second | Three quantities for the same rate and duration |
| Uniform | uniform_mean, uniform_conditional | Unconditional and conditional total mean |
| Binomial | binomial | r=0, 1, interior, n−1, n for one n,p story |
| Confidence interval | interval | Upper endpoint after changing sample size |

Diversity targets mathematical structure: signs, fractions, boundaries, units, ordering, nonzero supports, and interval multipliers. Shared direct templates are reported honestly; they are not evidence of unrestricted language generalization. Complete contrast groups must be checked individually, not inferred from aggregate row counts.

## Data and training

720 new training examples and 720 materialized historical training replay examples alternate 1:1. Development and final sets each contain 180 questions, ten per category. Historical retention evaluation is separate from actual training replay. Development and final rows must never enter training or teacher requests.

Teacher outputs retain raw provenance. Accepted teacher expressions and canonical corrections are distinguished; corrected targets must not be counted as accepted raw teacher responses. Exact arithmetic, semantic checks, structural equivalence, and negative formula mutations validate targets. Numerical coincidence alone is insufficient.

The prescribed run uses seed 2027, learning rate 1e-5, one epoch, batch size 1, accumulation 8, and checkpoints at updates 60, 120, and 180. This does not establish training-seed robustness.

## Historical gate and later evaluation

The original v15-based selection required non-decreasing category scores, zero paired losses, at least ten additional development answers, retention checks, and zero unresolved pending answers. No checkpoint passed every gate. Recorded deviations include a late baseline, out-of-order checkpoint inspection, and one post-output provisional credit.

The owner later requested a separate final evaluation against the original 3B Instruct base. That comparison yielded 45/180 versus 170/180. It demonstrates the bounded improvement, while leaving the original protocol result intact. See the [final report](STATS_DIVERSE_FINAL_BLIND_RESULTS.md).

## Original contracts and artifacts

- [Full curriculum protocol, Traditional Chinese](STATS_DIVERSE_CURRICULUM_PROTOCOL.md)
- [Full historical experiment protocol, Traditional Chinese](STATS_DIVERSE_EXPERIMENT_PROTOCOL.md)
- [Curriculum JSON](STATS_DIVERSE_CURRICULUM.json)
- [Materialized replay JSON](STATS_DIVERSE_REPLAY.json)
- [Legacy retention definition](STATS_DIVERSE_LEGACY_RETENTION.json)
- [Final question set, now exposed](STATS_DIVERSE_FINAL_BLIND.json)

These source contracts and data have not been translated or altered by this documentation update.
