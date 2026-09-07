"""Produce machine-readable offline readiness evidence before paid execution."""
import json
from pathlib import Path

from prepare_stats_consolidation_teacher import validate_only
from stats_consolidation_holdout import build as build_holdout
from stats_consolidation_pilot import DOCS, build, digest, selection_validation

REPO = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((DOCS / name).read_text())


def main():
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
    checks = {"generated_files_reproducible": stored == expected,
              "protocol_frozen": decision["protocol_status"] == "frozen_for_execution",
              "selection_frozen": selection["status"] == "frozen_for_execution",
              "selection_112": sum(len(x) for x in selection["suites"].values()) == 112,
              "holdout_288": sum(len(x) for x in holdout["suites"].values()) == 288,
              "teacher_benchmark_24": len(benchmark) == 24,
              "teacher_validate_only": validate_only()["questions"] == 268,
              "train_does_not_import_holdout": "stats_consolidation_holdout" not in (REPO / "scripts/run_stats_consolidation_compare.py").read_text(),
              "teacher_does_not_import_holdout": "stats_consolidation_holdout" not in (REPO / "scripts/prepare_stats_consolidation_teacher.py").read_text()}
    result = {"status": "ready_for_paid_pilot" if all(checks.values()) else "blocked",
              "checks": checks, "hashes": {key: digest(value) for key, value in stored.items()},
              "teacher_calls": 0, "gpu_runs": 0,
              "next": "paid teacher pilot; full generation only if pilot gate passes"}
    (DOCS / "STATS_CONSOLIDATION_READINESS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if result["status"] != "ready_for_paid_pilot":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
