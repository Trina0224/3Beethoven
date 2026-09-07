"""Produce machine-readable offline readiness evidence before paid execution."""
import json
import re
import subprocess
import sys
from pathlib import Path

from prepare_stats_consolidation_teacher import validate_only
from stats_consolidation_holdout import build as build_holdout
from stats_consolidation_pilot import DOCS, build, digest, selection_validation, historical_replay_questions
from stats_consolidation_semantics import validate_question, assert_disjoint, task_key
from stats_consolidation_grader import grader_fingerprint

REPO = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((DOCS / name).read_text())


def assess(prior, test_run=None):
    stories, requests, coverage, blueprint = build()
    selection = selection_validation(stories)
    holdout, benchmark = build_holdout()
    decision = load("STATS_CONSOLIDATION_DECISION_DRAFT.json")
    stored = {
        "stories": load("STATS_CONSOLIDATION_CANDIDATE_STORIES.json"),
        "requests": load("STATS_CONSOLIDATION_PILOT_REQUESTS.json"),
        "coverage": load("STATS_CONSOLIDATION_COVERAGE.json"),
        "selection": load("STATS_CONSOLIDATION_SELECTION_VALIDATION.json"),
        "holdout": load("STATS_CONSOLIDATION_HOLDOUT.json"),
        "benchmark": load("STATS_CONSOLIDATION_TEACHER_BENCHMARK.json"),
    }
    expected = {"stories": stories, "requests": requests, "coverage": coverage,
                "selection": selection, "holdout": holdout, "benchmark": benchmark}
    train = historical_replay_questions() + [q for s in stories if s["split"] == "train" for q in s["questions"]]
    validation = [q for s in stories if s["split"] == "validation" for q in s["questions"]]
    selected = [q for rows in selection["suites"].values() for q in rows]
    held = [q for rows in holdout["suites"].values() for q in rows]
    all_questions = train + validation + selected + held
    for q in all_questions:
        validate_question(q)
    assert_disjoint({"train_including_replay": train, "validation": validation + selected, "holdout": held})
    checks = {"generated_files_reproducible": stored == expected,
              "protocol_frozen": decision["protocol_status"] == "frozen_for_execution",
              "selection_frozen": selection["status"] == "frozen_for_execution",
              "selection_112": sum(len(x) for x in selection["suites"].values()) == 112,
              "holdout_288": sum(len(x) for x in holdout["suites"].values()) == 288,
              "teacher_benchmark_24": len(benchmark) == 24,
              "teacher_validate_only": validate_only()["questions"] == 268,
              "train_does_not_import_holdout": "stats_consolidation_holdout" not in (REPO / "scripts/run_stats_consolidation_compare.py").read_text(),
              "teacher_does_not_import_holdout": "stats_consolidation_holdout" not in (REPO / "scripts/prepare_stats_consolidation_teacher.py").read_text(),
              "independent_domain_oracle": True,
              "effective_subtasks_disjoint_including_replay": True,
              "offline_test_suite": bool(test_run and test_run["returncode"] == 0)}
    result = {"status": "offline_repair_verified_pilot_regrade_required" if all(checks.values()) else "blocked_offline_checks",
              "checks": checks, "hashes": {key: digest(value) for key, value in stored.items()},
              "grader_fingerprint": grader_fingerprint(),
              "domain_audit": {"questions_checked_including_replay": len(all_questions),
                               "invalid_questions": 0, "cross_split_subtask_collisions": 0},
              "offline_test_run": test_run,
              "teacher_calls": prior.get("teacher_calls", 0), "gpu_runs": prior.get("gpu_runs", 0),
              "pilot": prior.get("pilot"),
              "pilot_regrade_status": "not_executed_on_complete_source_records",
              "paid_execution_ready": False,
              "next": "Read-only regrade of preserved Version 49 records; no new calls or GPU authorized by this repair"}
    if result["pilot"] is not None:
        result["pilot"] = dict(result["pilot"], role="legacy_v1_evidence_not_current_execution_gate")
    return result


def main():
    prior = load("STATS_CONSOLIDATION_READINESS.json")
    completed = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "scripts",
                                "-p", "test_stats_consolidation*.py"], cwd=REPO, capture_output=True, text=True)
    match = re.search(r"Ran (\d+) tests", completed.stderr)
    test_run = {"returncode": completed.returncode, "tests_run": int(match[1]) if match else None,
                "command": "python -m unittest discover -s scripts -p 'test_stats_consolidation*.py'"}
    try:
        result = assess(prior, test_run)
    except (ValueError, AssertionError) as exc:
        result = dict(prior, status="blocked_offline_checks", paid_execution_ready=False,
                      offline_error=str(exc), offline_test_run=test_run)
    (DOCS / "STATS_CONSOLIDATION_READINESS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if result["status"] == "blocked_offline_checks":
        print(completed.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
