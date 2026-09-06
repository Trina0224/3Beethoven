## Grader correction: 2026-09-06

Regraded the existing 320 teacher responses without new API calls. Mathematical
formulation and output formatting now have separate scores. Missing labels alone
no longer fail a correct expression. Names are substituted only from the teacher's
own assignments; reference bindings never repair the response. Automatic credit
requires exact execution and a reviewed formula structure, not just answer equality.
The selected final expression is assessed, not every claim in the reasoning.

- First response: **151/160 (94.375%)** verified.
- At least one of the two saved responses: **157/160 (98.125%)** verified.
- Across all 320 attempts: **302 verified, 15 incorrect, 3 pending review**.
- Training candidates: 126/128 have a verified response; validation: 31/32.

This is a post-hoc grader correction on preparation data, not a passed independent
perturbation test or a student training result. Original raw responses and old scores
remain unchanged. Unsupported equivalent structures remain pending instead of being
called mathematically wrong. Answer-only outputs do not establish correct formulation.

Reproduce with `python scripts/regrade_teacher_v0_14.py`.
Results: [teacher regrade](STATS_V0_14_TEACHER_REGRADED.json).
The offline regrade is authoritative for existing records; cached preparation records
retain their historical acceptance flags. v0.14's scorer uses the corrected grader for
new responses. No paid preparation rerun is needed. Student v0.14 training has not started.

