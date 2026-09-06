# v0.15 paired diagnostic

Status: running after completed v0.15 training; no new training or teacher calls.

Purpose: distinguish reading/binding, affine mean, affine variance and second-moment composition failures. Compare saved v0.14 and v0.15 adapters on the same 16 previously tested stories (8 moment, 8 affine Poisson variance). This is a post-test diagnostic, not a new held-out benchmark or a replacement for the recorded 44/64 result.

Each story has eight independent prompts: extract mean, extract variance, extract multiplier, extract offset, formulate transformed mean, formulate transformed variance, original task, and original task with generic symbolic reminders. Each response begins a new conversation; no correct subanswer is passed into a later prompt. There are 128 prompts per checkpoint, 256 responses total. The original-task condition uses the diagnostic output instruction, so it is not a byte-identical repeat of the earlier evaluation prompt.

References and prompts are fixed by `scripts/diagnose_stats_v0_15.py` before execution. Existing safe expression grading is used; all 128 reference expressions, wrong-zero responses and unresolved names were checked locally. For extraction a scalar is appropriate; for formulation a matching number alone remains pending semantic review. Review pending outputs before interpreting counts. A failed formatting response is not proof that the underlying fact is unknown.

The hinted condition explicitly supplies general mathematical rules and must remain separate from unaided performance. These probes can identify patterns but cannot by themselves establish the causal reason a model failed. No checkpoint selection uses these results.
