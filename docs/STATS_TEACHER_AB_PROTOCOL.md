# Teacher-format development comparison — preregistration

Recorded 2026-09-07 before new teacher calls. User authorized autonomous continuation while unavailable. This is a development experiment, not a student result or a held-out teacher benchmark.

## Motivation and expectations

V49 mixed six providers and retried mostly identical temperature-zero requests. Offline and production parsers diverged. Shared conservative parsing recovers 59/65 targets, including an echoed nested answer, but six mathematical failures remain. These are results already visible before this comparison; the original V49 reports remain unchanged. The new fingerprint invalidates old execution gates.

Hypothesis: a one-question request with a fixed provider and explicit, independently checkable intermediate quantities can improve dependable generation. A compares direct expression; B requests named intermediate quantities then the final expression. B is **scaffolding plus output-format intervention**, not a pure formatting ablation. Neither arm receives a correct answer. Both arms use identical question wording. Failure of B on the same concepts weakens this hypothesis. A/B cannot identify why V49 failed because grouping/provider also differ from V49.

## Frozen design

- 24 new development questions: eight Poisson unit/count-variance, eight interval upper endpoints, eight affine-Poisson second moments. Fixed generation seed 210907. Reject semantic collisions with existing replay, candidate, selection and sealed holdout. Development questions never become training or final evaluation examples.
- Model meta-llama/llama-3.3-70b-instruct; provider DeepInfra only, fallbacks disabled; temperature 0; JSON-object response, maximum 400 output tokens. No schema-support assumption. Record request hash, raw answer, model, provider, finish reason, usage and cost.
- Each question once per arm: 48 calls, alternating AB/BA order. No automatic retries. Any unresolved request or missing cost stops spending. Save every response immediately.
- Cost ceiling $0.50 including prior V49 reported $0.00402078, with $0.01 reserved per call. Global experiment limits remain $3 and 216 calls including prior 38 and these 48. No increase of limits.
- Primary arm acceptance: final expression passes the frozen numerical/structural grader; B additionally requires every requested intermediate to equal its independent rational oracle. Reject placeholders, duplicate keys, ambiguous envelopes, incomplete answers, wrong model/provider and non-stop finish reasons. Arithmetic steps may be evaluated, final expression remains unevaluated.
- An arm qualifies only at >=22/24 overall and >=7/8 in every family, with no unresolved/pending reviews. Choose higher accepted count among qualifying arms; ties prefer simpler A. Report paired flips and individual failures, not population-level certainty from 24 questions.
- If neither qualifies: save evidence, do not generate a full corpus or train. Reference-assisted generation/model changes require a new explicit research decision.
- If one qualifies: integrate that request contract into fresh corpus generation, first freeze genuine paraphrases, preserve all original corpus/retention gates and two starts x two seeds. A passing development screen alone does not release GPU training. All students still target one-line expressions; intermediate checks are teacher-only.

## Scope and stop conditions

No adapter merging, no new teacher model, no second training epoch, no test-based checkpoint selection. Permissions, unknown charges, failed provider pin or data/quality gates stop execution with preserved evidence. This preregistration must be committed before the first new call; outcomes are appended separately.
