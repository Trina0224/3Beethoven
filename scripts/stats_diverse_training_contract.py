"""CPU-verifiable training and promotion contract for the diverse student."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURRICULUM_PATH = ROOT / "docs" / "STATS_DIVERSE_CURRICULUM.json"
REPLAY_PATH = ROOT / "docs" / "STATS_DIVERSE_REPLAY.json"


class TrainingContractError(RuntimeError):
    pass


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def promotion_credit(judged):
    """The only mathematical credit rule; format flags are diagnostics only."""
    if not isinstance(judged, dict):
        raise TrainingContractError("Grader result must be an object")
    return judged.get("math_correct") is True and judged.get("executable") is True


def promotion_vector(evaluated_rows):
    return [promotion_credit(row["grader"]) for row in evaluated_rows]


def format_vector(evaluated_rows):
    return [
        {
            "whole_raw_executable": row["grader"].get("whole_raw_executable") is True,
            "strict_one_line_expression": row["grader"].get("strict_one_line_expression") is True,
        }
        for row in evaluated_rows
    ]


def interleave_training_rows(new_rows, replay_rows):
    """Materialize a fixed 1:1 rehearsal order, never an evaluation-only gate."""
    if not isinstance(new_rows, list) or not isinstance(replay_rows, list):
        raise TrainingContractError("Training inputs must be row lists")
    if len(new_rows) != 720 or len(replay_rows) != 720:
        raise TrainingContractError("The frozen run requires 720 new and 720 replay rows")
    if any(row.get("split") != "train" for row in new_rows):
        raise TrainingContractError("New training input contains a non-train row")
    if any(row.get("source_split") != "train" or row.get("training_role") != "legacy_replay"
           for row in replay_rows):
        raise TrainingContractError("Replay input contains a non-training source row")
    if {row["id"] for row in new_rows} & {row["id"] for row in replay_rows}:
        raise TrainingContractError("New and replay IDs overlap")
    ordered = []
    for new, replay in zip(new_rows, replay_rows):
        ordered.append({"training_role": "new_distillation", "row": new})
        ordered.append({"training_role": "legacy_replay", "row": replay})
    roles = [item["training_role"] for item in ordered]
    if roles != [role for _ in range(720) for role in ("new_distillation", "legacy_replay")]:
        raise TrainingContractError("Replay interleave drift")
    return ordered


def load_training_plan(curriculum_path=CURRICULUM_PATH, replay_path=REPLAY_PATH):
    curriculum = json.loads(Path(curriculum_path).read_text(encoding="utf-8"))
    replay = json.loads(Path(replay_path).read_text(encoding="utf-8"))
    new_rows = curriculum.get("train")
    replay_rows = replay.get("replay")
    if curriculum.get("manifest", {}).get("counts", {}).get("train") != 720:
        raise TrainingContractError("Curriculum train receipt mismatch")
    replay_manifest = replay.get("manifest", {})
    if replay_manifest.get("count") != 720 or replay_manifest.get("rows_sha256") != digest(replay_rows):
        raise TrainingContractError("Replay receipt mismatch")
    ordered = interleave_training_rows(new_rows, replay_rows)
    category_counts = Counter(item["row"]["category"] for item in ordered)
    if set(category_counts.values()) != {80}:
        raise TrainingContractError("Combined training plan is not category-balanced")
    receipt = {
        "new_count": 720,
        "replay_count": 720,
        "total_count": 1440,
        "new_to_replay_ratio": "1:1",
        "alternating_roles": True,
        "category_counts": dict(sorted(category_counts.items())),
        "ordered_ids_sha256": digest(
            [(item["training_role"], item["row"]["id"]) for item in ordered]
        ),
    }
    return ordered, receipt
