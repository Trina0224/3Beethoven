"""Build the historical training replay used by the diverse student.

Retention development/test rows remain evaluation-only. Replay is selected
solely from the frozen final_clear training split and is materialized with its
full prompt and target.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from prepare_stats_diverse_legacy import (
    DEFAULT_SOURCE,
    EXPECTED_CATEGORIES,
    canonical_sha256,
    load_source,
)
from stats_consolidation_grader import grader_fingerprint, score
from stats_consolidation_semantics import validate_question


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "STATS_DIVERSE_REPLAY.json"
REPLAY_REVISION = "stats-diverse-replay-v1"
SELECTION_SEED = 91357
PER_CATEGORY = 40


class ReplayContractError(RuntimeError):
    pass


def selection_key(row):
    material = f"{REPLAY_REVISION}|{SELECTION_SEED}|{row['id']}".encode()
    return hashlib.sha256(material).hexdigest()


def build(source=DEFAULT_SOURCE):
    document, source_hashes = load_source(source)
    rows = document.get("train")
    if not isinstance(rows, list):
        raise ReplayContractError("Frozen source has no training split")
    declared = document.get("manifest", {}).get("split_sha256", {}).get("train")
    actual = canonical_sha256(rows)
    if declared != actual:
        raise ReplayContractError("Historical training split hash drift")

    by_category = defaultdict(list)
    seen_ids = set()
    for row in rows:
        if row.get("id") in seen_ids:
            raise ReplayContractError("Duplicate historical training ID")
        seen_ids.add(row.get("id"))
        category = row.get("category")
        if category not in EXPECTED_CATEGORIES:
            raise ReplayContractError("Unexpected historical training category")
        try:
            validate_question(row)
            judged = score(row.get("target"), row)
        except Exception as exc:
            raise ReplayContractError(
                f"Historical training row failed validation: {row.get('id')}: {exc}"
            ) from None
        if not (
            judged.get("primary_correct") is True
            and judged.get("review_required") is False
            and judged.get("strict_one_line_expression") is True
        ):
            raise ReplayContractError(
                f"Historical training target is not replay-safe: {row['id']}"
            )
        by_category[category].append(row)

    selected = []
    for category in EXPECTED_CATEGORIES:
        candidates = sorted(by_category[category], key=selection_key)
        if len(candidates) < PER_CATEGORY:
            raise ReplayContractError(f"Not enough replay rows for {category}")
        for source_row in candidates[:PER_CATEGORY]:
            row = copy.deepcopy(source_row)
            row["training_role"] = "legacy_replay"
            row["source_corpus"] = "STATS_FINAL_CLEAR_DATA.json.gz.b64"
            row["source_split"] = "train"
            row["source_row_sha256"] = canonical_sha256(source_row)
            selected.append(row)

    buckets = {
        category: [row for row in selected if row["category"] == category]
        for category in EXPECTED_CATEGORIES
    }
    ordered = [
        buckets[category][index]
        for index in range(PER_CATEGORY)
        for category in EXPECTED_CATEGORIES
    ]
    manifest = {
        "schema_version": REPLAY_REVISION,
        "status": "frozen_before_student_training",
        "purpose": "actual training replay; never evaluation",
        "source": {
            "repository_path": "docs/STATS_FINAL_CLEAR_DATA.json.gz.b64",
            **source_hashes,
            "source_train_rows_sha256": actual,
        },
        "selection_seed": SELECTION_SEED,
        "selection_method": "sha256(revision|seed|source_row_id), first 40 per category",
        "count": len(ordered),
        "per_category": PER_CATEGORY,
        "category_counts": dict(
            sorted(Counter(row["category"] for row in ordered).items())
        ),
        "row_ids_sha256": canonical_sha256([row["id"] for row in ordered]),
        "rows_sha256": canonical_sha256(ordered),
        "rows_materialized": True,
        "evaluation_rows_included": False,
        "required_new_to_replay_ratio": "1:1",
        "grader_fingerprint": grader_fingerprint(),
    }
    return {"replay": ordered, "manifest": manifest}


def render(value):
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render(build(args.source))
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != expected:
            raise ReplayContractError("Frozen replay artifact is missing or stale")
    else:
        args.output.write_text(expected, encoding="utf-8")
    print(hashlib.sha256(expected.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
