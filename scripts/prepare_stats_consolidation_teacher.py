"""Resumable teacher preparation for the consolidation candidate.

No call is possible while the decision file remains a draft.  References stay
outside requests; accepted expressions are grader-checked teacher outputs.
"""
import argparse
import json
import threading
from pathlib import Path

from flight_run_stats_v0_3 import TeacherClient, package, read_json, save_json
from stats_curriculum_v0_19 import score
from stats_consolidation_pilot import DOCS, TEACHER_MODEL, build, digest

ROOT = Path("/kaggle/working/3beethoven_stats_consolidation_teacher")
DECISION = DOCS / "STATS_CONSOLIDATION_DECISION_DRAFT.json"


class BudgetedClient(TeacherClient):
    def __init__(self, key, max_calls, cost_cap):
        ROOT.mkdir(parents=True, exist_ok=True)
        super().__init__(ROOT, key, max_calls)
        self.cost_cap = cost_cap
        self.lock = threading.Lock()

    def call(self, tag, *args, **kwargs):
        with self.lock:
            cached = (self.root / "api_cache" / (tag + ".json")).exists()
            usage = self.stats()
            if not cached and (usage["reported_cost_usd"] >= self.cost_cap or usage["responses_without_cost"]):
                raise RuntimeError("Teacher cost gate reached or a completed response has no reported cost")
        return super().call(tag, *args, **kwargs)


def parse_answers(raw):
    value = json.loads(raw)
    answers = value.get("answers")
    if not isinstance(answers, list):
        raise ValueError("answers must be a list")
    result = {}
    for item in answers:
        if not isinstance(item, dict) or set(item) != {"question_id", "expression"}:
            raise ValueError("Each answer must contain only question_id and expression")
        if not isinstance(item["question_id"], str) or not isinstance(item["expression"], str):
            raise ValueError("Answer fields must be strings")
        if item["question_id"] in result:
            raise ValueError("Duplicate question_id")
        result[item["question_id"]] = item["expression"].strip()
    return result


def request_messages(request, pending_ids):
    questions = [q for q in request["questions"] if q["question_id"] in pending_ids]
    user = {
        "story_id": request["story_id"],
        "questions": questions,
        "required_response": {"answers": [{"question_id": q["question_id"], "expression": "<expression>"} for q in questions]},
    }
    system = (
        "Solve each statistics question independently. Return only one JSON object matching the supplied schema. "
        "Each expression must be fully numerical and executable using integers, fractions, +, -, *, /, **, "
        "and comb(n,r). Preserve unit conversions, complements and powers as operations. Do not evaluate the "
        "expression, add prose, introduce variables, or omit requested questions."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, separators=(",", ":"))}]


def build_verified(stories):
    rows = {"train": [], "validation": []}
    for story in stories:
        record = read_json(ROOT / "records" / (story["story_id"] + ".json"), {})
        accepted = record.get("accepted", {})
        for q in story["questions"]:
            item = accepted.get(q["id"])
            if not item:
                continue
            rows[story["split"]].append({
                "source_id": q["id"], "story_id": story["story_id"], "lineage_id": story["lineage_id"],
                "category": q["category"], "prompt": q["question"] + "\nReturn one line: Expression: <a fully substituted numerical expression>. Keep arithmetic and unit conversions unevaluated. No final answer or explanation.",
                "target": "Expression: " + item["normalized_expression"],
                "teacher_raw": item["teacher_expression"], "teacher_model": TEACHER_MODEL,
                "question_sha256": digest(q), "validation": item["judged"],
                "reference_conditioned": False,
            })
    return rows


def validate_only():
    stories, requests, coverage, holdout = build()
    assert len(stories) == len(requests) == 96 and coverage["pilot_story_count"] == 24
    assert holdout["status"] == "blueprint_only_no_concrete_questions"
    for request in requests:
        pending = {q["question_id"] for q in request["questions"]}
        messages = request_messages(request, pending)
        assert len(messages[1]["content"].encode()) <= 4000
        assert not any(key in messages[1]["content"] for key in ('"answer"', '"bindings"', '"reference"'))
    return {"stories": len(stories), "questions": coverage["question_count"], "pilot": 24,
            "max_request_bytes": max(len(request_messages(r, {q["question_id"] for q in r["questions"]})[1]["content"].encode()) for r in requests)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("validate-only", "pilot", "full"), default="validate-only")
    args = parser.parse_args()
    check = validate_only()
    if args.scope == "validate-only":
        print(json.dumps(check)); return
    decision = read_json(DECISION)
    if decision.get("protocol_status") != "frozen_for_execution":
        raise RuntimeError("Decision file is not frozen_for_execution; paid teacher calls are blocked")
    teacher = decision["teacher"]
    if teacher["model"] != TEACHER_MODEL or teacher["temperature"] != 0 or teacher["max_attempts_per_story"] != 2:
        raise RuntimeError("Teacher identity or generation contract changed")
    max_calls = teacher["pilot_max_calls"] if args.scope == "pilot" else teacher["full_max_calls"]
    cost_cap = teacher["pilot_cost_cap_usd"] if args.scope == "pilot" else teacher["full_total_cost_cap_usd"]
    from kaggle_secrets import UserSecretsClient
    client = BudgetedClient(UserSecretsClient().get_secret("OPENROUTER_API_KEY"), max_calls, cost_cap)
    stories, requests, coverage, _ = build()
    story_lookup = {s["story_id"]: s for s in stories}
    selected = [r for r in requests if args.scope == "full" or r["pilot_batch"]]
    save_json(ROOT / "run_contract.json", {"scope": args.scope, "decision_sha256": digest(decision),
              "request_sha256": digest(requests), "teacher_model": TEACHER_MODEL,
              "max_calls": max_calls, "cost_cap_usd": cost_cap})
    try:
        for position, request in enumerate(selected, 1):
            path = ROOT / "records" / (request["story_id"] + ".json")
            record = read_json(path, {"story_id": request["story_id"], "request_sha256": digest(request), "attempts": [], "accepted": {}})
            if record["request_sha256"] != digest(request):
                raise RuntimeError("Request changed for existing story record")
            story = story_lookup[request["story_id"]]
            qlookup = {q["id"]: q for q in story["questions"]}
            for attempt in range(len(record["attempts"]), 2):
                pending = set(qlookup) - set(record["accepted"])
                if not pending:
                    break
                raw = client.call(request["story_id"] + f"_attempt_{attempt}", request_messages(request, pending),
                                  max_tokens=teacher["max_tokens_per_story"], json_mode=True)
                attempt_record = {"attempt": attempt, "raw": raw, "parsed": None, "error": None}
                try:
                    parsed = parse_answers(raw)
                    attempt_record["parsed"] = parsed
                    for qid in pending:
                        if qid not in parsed:
                            continue
                        judged = score("Expression: " + parsed[qid], qlookup[qid])
                        if judged["math_correct"] is True and judged["executable"]:
                            record["accepted"][qid] = {"teacher_expression": parsed[qid], "normalized_expression": judged["normalized_expression"], "judged": judged, "attempt": attempt}
                except (ValueError, json.JSONDecodeError) as exc:
                    attempt_record["error"] = type(exc).__name__ + ": " + str(exc)
                record["attempts"].append(attempt_record)
                save_json(path, record)
            print("TEACHER", position, "/", len(selected), request["story_id"], len(record["accepted"]), "/", len(qlookup), json.dumps(client.stats()), flush=True)
        verified = build_verified(stories)
        save_json(ROOT / "verified_teacher_rows.json", verified)
        save_json(ROOT / "summary.json", {"scope": args.scope, "coverage_sha256": digest(coverage),
                  "accepted": {k: len(v) for k, v in verified.items()}, "usage": client.stats(),
                  "teacher_model": TEACHER_MODEL, "reference_conditioned": False})
    finally:
        save_json(ROOT / "api_usage.json", client.stats())
        package(ROOT)


if __name__ == "__main__":
    main()
