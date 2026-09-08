"""Freeze the already-public V15-era development/test rows as retention gates.

The source bundle stays the single copy of the rows.  This script validates all
216 references with the current semantics/oracle/grader and emits only a small,
externally pinnable manifest.  The student runner can then load ``development``
before training and defer use of ``test`` until the authorized final phase.
"""
from __future__ import annotations

import argparse
import base64
import collections
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from stats_consolidation_grader import grader_fingerprint, score
from stats_consolidation_semantics import validate_question


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "docs" / "STATS_FINAL_CLEAR_DATA.json.gz.b64"
DEFAULT_OUTPUT = ROOT / "docs" / "STATS_DIVERSE_LEGACY_RETENTION.json"
SOURCE_FILE_SHA256 = "efe223810a704455d234f388c2f1d74e001a16cdd079701319295d957c2759d1"
SOURCE_GZIP_SHA256 = "63f6ae30240c5f3b5b97e23bc91221ab66b34d1965c25188ac572b57c25c40c4"
SOURCE_JSON_SHA256 = "c5ee6f022226469bd33c51dc95455ae8d130c07dc332d6dad9c084b27976f2c1"
EXPECTED_CATEGORIES = (
    "both",
    "neither",
    "exactly_one",
    "at_least_one",
    "same",
    "moment_mean",
    "moment_variance",
    "moment_second",
    "poisson_variance",
    "poisson_scaled",
    "poisson_second",
    "process_variance",
    "process_scaled",
    "process_second",
    "uniform_mean",
    "uniform_conditional",
    "binomial",
    "interval",
)
SPLITS = {
    "development": {
        "source_name": "development",
        "count": 72,
        "per_category": 4,
        "sha256": "52bf46dc3efe2a69966be57c0573e14dad68c52b832b3d31c76b646fc5f4c54c",
        "order_sha256": "c183c543341a56e16e96705a825734d3bd637a2a2031b0dcb750be457364207c",
    },
    "final": {
        "source_name": "test",
        "count": 144,
        "per_category": 8,
        "sha256": "852e5fe5feb34607e71c7adda3f24bf156a0cbdbc1f3f6ca3126372ae754f003",
        "order_sha256": "0384c5e822e98d24b9c596d5885e7e5ca86bf316598d0da43764b6ea766a359f",
    },
}


class LegacyContractError(RuntimeError):
    """The historical retention source no longer matches its frozen contract."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256_bytes(encoded)


def load_source(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    encoded = path.read_bytes()
    if sha256_bytes(encoded) != SOURCE_FILE_SHA256:
        raise LegacyContractError("Historical source-file SHA-256 drift")
    try:
        gzip_bytes = base64.b64decode(b"".join(encoded.split()), validate=True)
        decoded = gzip.decompress(gzip_bytes)
        document = json.loads(decoded)
    except Exception as exc:
        raise LegacyContractError(f"Historical source cannot be decoded: {exc}") from None
    if sha256_bytes(gzip_bytes) != SOURCE_GZIP_SHA256:
        raise LegacyContractError("Historical gzip-payload SHA-256 drift")
    if sha256_bytes(decoded) != SOURCE_JSON_SHA256:
        raise LegacyContractError("Historical decoded-JSON SHA-256 drift")
    if not isinstance(document, dict):
        raise LegacyContractError("Historical decoded source must be an object")
    return document, {
        "file_sha256": SOURCE_FILE_SHA256,
        "gzip_payload_sha256": SOURCE_GZIP_SHA256,
        "decoded_json_sha256": SOURCE_JSON_SHA256,
    }


def validate_split(
    rows: Any, public_name: str, contract: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise LegacyContractError(f"{public_name} must be a list of objects")
    if len(rows) != contract["count"]:
        raise LegacyContractError(f"{public_name} row-count drift")
    if canonical_sha256(rows) != contract["sha256"]:
        raise LegacyContractError(f"{public_name} canonical SHA-256 drift")
    if canonical_sha256([row.get("id") for row in rows]) != contract["order_sha256"]:
        raise LegacyContractError(f"{public_name} row-order SHA-256 drift")

    ids = [row.get("id") for row in rows]
    if any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
        raise LegacyContractError(f"{public_name} IDs are missing or duplicated")
    categories = collections.Counter(row.get("category") for row in rows)
    expected_counts = {name: contract["per_category"] for name in EXPECTED_CATEGORIES}
    if dict(categories) != expected_counts:
        raise LegacyContractError(f"{public_name} category balance drift: {dict(categories)}")

    for row in rows:
        try:
            validate_question(row)
            judged = score(row.get("target"), row)
        except Exception as exc:
            raise LegacyContractError(
                f"Historical row {row.get('id', '<missing>')} failed validation: {exc}"
            ) from None
        if judged.get("primary_correct") is not True or judged.get("review_required") is not False:
            raise LegacyContractError(
                f"Historical reference {row['id']} is not an unambiguous grader pass"
            )
    return {
        "source_split": contract["source_name"],
        "count": len(rows),
        "rows_sha256": canonical_sha256(rows),
        "order_sha256": canonical_sha256(ids),
        "category_counts": dict(sorted(categories.items())),
        "reference_targets_accepted": len(rows),
        "reference_targets_pending": 0,
    }


def build(source: Path = DEFAULT_SOURCE) -> dict[str, Any]:
    document, source_hashes = load_source(source)
    source_manifest = document.get("manifest")
    if not isinstance(source_manifest, dict):
        raise LegacyContractError("Historical source manifest is missing")

    split_receipts = {}
    for public_name, contract in SPLITS.items():
        source_name = contract["source_name"]
        rows = document.get(source_name)
        split_receipts[public_name] = validate_split(rows, public_name, contract)
        declared = source_manifest.get("split_sha256", {}).get(source_name)
        if declared != contract["sha256"]:
            raise LegacyContractError(f"Historical source manifest drift for {source_name}")

    return {
        "schema_version": "stats-diverse-legacy-retention-v1",
        "status": "frozen_before_student_training",
        "purpose": "V15-era capability retention only; not evidence of new-scope improvement",
        "source": {
            "repository_path": "docs/STATS_FINAL_CLEAR_DATA.json.gz.b64",
            **source_hashes,
        },
        "categories": list(EXPECTED_CATEGORIES),
        "splits": split_receipts,
        "validation": {
            "grader_fingerprint": grader_fingerprint(),
            "all_216_reference_targets_accepted": True,
            "pending_equivalences": 0,
            "rows_duplicated_into_this_manifest": False,
        },
    }


def render(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    expected = render(build(args.source))
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != expected:
            raise LegacyContractError(f"Frozen legacy manifest is missing or stale: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(expected, encoding="utf-8")
    print(sha256_bytes(expected.encode("utf-8")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
