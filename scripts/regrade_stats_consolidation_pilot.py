"""Read-only source regrading. No network client, training, or source writes.

An audit does not release the paid/full gate. Use a fresh output filename.
Changed request hashes are reported as stale rather than reusing old answers.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from stats_consolidation_grader import score, outcome, grader_fingerprint
from stats_consolidation_pilot import build, digest


def parse_answers(raw):
    """Parse the teacher envelope, repairing only redundant trailing braces.

    The original bytes remain in the source record.  We intentionally do not
    recover answers nested in an echoed request envelope.
    """
    repair = None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        stripped = raw.strip()
        payload, end = json.JSONDecoder().raw_decode(stripped)
        trailing = stripped[end:].strip()
        if not trailing or set(trailing) != {"}"}:
            raise
        repair = "removed_redundant_trailing_closing_brace"
    items = payload["answers"]
    if not isinstance(items, list):
        raise ValueError("answers must be a list")
    return items, repair


def audit(stories, requests, source):
    lookup = {s["story_id"]: s for s in stories}
    rows, request_errors = [], []
    for request in (r for r in requests if r["pilot_batch"]):
        story = lookup[request["story_id"]]
        path = source / "records" / (story["story_id"] + ".json")
        record = json.loads(path.read_text()) if path.exists() else None
        compatible = record is not None and record.get("request_sha256") == digest(request)
        if not compatible:
            reason = "missing_record" if record is None else "changed_request"
            request_errors.append({"story_id": story["story_id"], "reason": reason})
        for q in story["questions"]:
            attempts = []
            if compatible:
                for index, a in enumerate(record["attempts"]):
                    try:
                        items, transport_repair = parse_answers(a["raw"])
                        ids = [x["question_id"] for x in items]
                        if len(ids) != len(set(ids)):
                            raise ValueError("Duplicate question ID in raw JSON")
                        matches = [x["expression"] for x in items if x["question_id"] == q["id"]]
                        if not matches:
                            continue
                        judged = score("Expression: " + matches[0], q)
                        attempts.append({"attempt": index, "raw_expression": matches[0],
                                         "classification": outcome(judged), "judged": judged,
                                         "transport_repair": transport_repair})
                    except (ValueError, KeyError, TypeError) as exc:
                        attempts.append({"attempt": index, "classification": "malformed_response",
                                         "error": str(exc)})
            accepted = next((a for a in attempts if a["classification"] == "accepted"), None)
            status = "accepted" if accepted else attempts[-1]["classification"] if attempts else "unavailable"
            rows.append({"id": q["id"], "archetype": story["archetype"], "category": q["category"],
                         "question": q["question"], "source_request_compatible": compatible,
                         "originally_accepted": q["id"] in (record or {}).get("accepted", {}),
                         "classification": status, "first_attempt_accepted": bool(attempts and
                          attempts[0]["attempt"] == 0 and attempts[0]["classification"] == "accepted"),
                         "attempts": attempts})
    counts = dict(Counter(r["classification"] for r in rows))
    return {"role": "offline_diagnostic_only_not_execution_gate", "teacher_calls_added": 0,
            "grader_fingerprint": grader_fingerprint(), "planned_targets": len(rows),
            "source_request_errors": request_errors, "classifications": counts,
            "first_attempt_accepted": sum(r["first_attempt_accepted"] for r in rows),
            "newly_credited": [r["id"] for r in rows if r["classification"] == "accepted" and not r["originally_accepted"]],
            "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source_root.resolve(), args.output.resolve()
    if not (source / "records").is_dir():
        raise SystemExit("Source records are missing; no results produced")
    if output.exists() or output.is_relative_to(source):
        raise SystemExit("Use a new output file outside the preserved source directory")
    stories, requests, _, _ = build()
    result = audit(stories, requests, source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


if __name__ == "__main__":
    main()
