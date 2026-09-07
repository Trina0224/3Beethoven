"""Isolated grader contract for the consolidation experiment.

Historical graders remain frozen.  This wrapper records the exact raw response,
keeps mathematical/executable/format judgements separate, and defines the only
primary-credit rule used by the new selection and holdout evaluators.
"""
import hashlib
import re

from stats_curriculum_v0_19 import score as historical_score

GRADER_VERSION = "stats-consolidation-v1"


def score(raw, question):
    if not isinstance(raw, str):
        raise TypeError("raw model response must be a string")
    judged = dict(historical_score(raw, question))
    judged["grader_version"] = GRADER_VERSION
    judged["raw_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    judged["strict_one_line_expression"] = bool(
        re.fullmatch(r"\s*Expression:\s*[^\n]+\s*", raw)
    )
    judged["primary_correct"] = bool(
        judged.get("math_correct") is True and judged.get("executable") is True
    )
    judged["correct"] = judged["primary_correct"]
    return judged

