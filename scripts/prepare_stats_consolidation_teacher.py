"""Resumable teacher preparation for the consolidation candidate.

No call is possible while the decision file remains a draft.  References stay
outside requests; accepted expressions are grader-checked teacher outputs.
"""
import argparse
import json
import threading
from pathlib import Path

from flight_run_stats_v0_3 import TeacherClient, package, read_json, save_json
from stats_consolidation_pilot import DOCS, TEACHER_MODEL, build, digest, assert_execution_released
from stats_consolidation_grader import score, outcome, grader_fingerprint
from stats_teacher_envelope import parse_answers as parse_envelope

ROOT = Path("/kaggle/working/3beethoven_stats_consolidation_teacher")
DECISION = DOCS / "STATS_CONSOLIDATION_DECISION_DRAFT.json"


class BudgetedClient(TeacherClient):
    def __init__(self, key, max_calls, cost_cap, reserve_per_call):
        ROOT.mkdir(parents=True, exist_ok=True)
        super().__init__(ROOT, key, max_calls)
        self.cost_cap = cost_cap
        self.reserve_per_call = reserve_per_call
        self.lock = threading.Lock()

    def call(self, tag, *args, **kwargs):
        with self.lock:
            cached = (self.root / "api_cache" / (tag + ".json")).exists()
            usage = self.stats()
            if not cached and (usage["reported_cost_usd"] + self.reserve_per_call > self.cost_cap
                               or usage["responses_without_cost"]):
                raise RuntimeError("Teacher cost gate reached or a completed response has no reported cost")
        return super().call(tag, *args, **kwargs)


def parse_answers(raw):
    answers, _ = parse_envelope(raw)
    return {item["question_id"]: item["expression"].strip() for item in answers}


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
                "teacher_raw_sha256": digest(item["teacher_expression"]),
                "reference_conditioned": False,
            })
    return rows


def acceptance_report(stories, scope):
    selected = [s for s in stories if scope == "full" or any(
        r["story_id"] == s["story_id"] and r["pilot_batch"] for r in build()[1]
    )]
    planned_archetype = {}
    accepted_archetype = {}
    planned_category = {}
    accepted_category = {}
    pending_ids = []
    unresolved_ids = []
    outcomes = {}
    for story in selected:
        record = read_json(ROOT / "records" / (story["story_id"] + ".json"), {})
        accepted = record.get("accepted", {})
        planned_archetype[story["archetype"]] = planned_archetype.get(story["archetype"], 0) + len(story["questions"])
        accepted_archetype[story["archetype"]] = accepted_archetype.get(story["archetype"], 0) + len(accepted)
        for q in story["questions"]:
            planned_category[q["category"]] = planned_category.get(q["category"], 0) + 1
            accepted_category[q["category"]] = accepted_category.get(q["category"], 0) + int(q["id"] in accepted)
            if q["id"] not in accepted:
                pending_ids.append(q["id"])
                parsed = next((a.get("parsed", {}) for a in reversed(record.get("attempts", []))
                               if a.get("parsed") and q["id"] in a["parsed"]), {})
                if q["id"] in parsed:
                    judged = score("Expression: " + parsed[q["id"]], q)
                    outcomes[q["id"]] = outcome(judged)
                    if judged["review_required"]:
                        unresolved_ids.append(q["id"])
                else:
                    outcomes[q["id"]] = "missing_answer"
                    unresolved_ids.append(q["id"])
    return {"scope": scope, "planned": sum(planned_archetype.values()),
            "accepted": sum(accepted_archetype.values()), "pending_ids": pending_ids,
            "planned_by_archetype": planned_archetype, "accepted_by_archetype": accepted_archetype,
            "planned_by_category": planned_category, "accepted_by_category": accepted_category,
            "unresolved_ids": unresolved_ids, "rejected_outcomes": outcomes}


def gate_acceptance(report, teacher):
    import math
    checks = {"no_unresolved_scoring": not report.get("unresolved_ids", report["pending_ids"])}
    if report["scope"] == "pilot":
        checks["total"] = report["accepted"] >= teacher["pilot_minimum_accepted_targets"]
        checks.update({"archetype:" + key: report["accepted_by_archetype"].get(key, 0) >= math.ceil(value * 0.8)
                       for key, value in report["planned_by_archetype"].items()})
    else:
        checks.update({"category:" + key: report["accepted_by_category"].get(key, 0) >= math.ceil(value * 0.8)
                       for key, value in report["planned_by_category"].items()})
    return {"passed": all(checks.values()), "checks": checks,
            "unresolved_count": len(report.get("unresolved_ids", report["pending_ids"])),
            "not_accepted_count": len(report["pending_ids"])}


def validate_only():
    stories, requests, coverage, holdout = build()
    assert len(stories) == len(requests) == 96 and coverage["pilot_story_count"] == 24
    assert holdout["status"] == "superseded_by_STATS_CONSOLIDATION_HOLDOUT.json"
    sealed = read_json(DOCS / "STATS_CONSOLIDATION_HOLDOUT.json")
    assert sealed["metadata"]["status"] == "sealed_after_domain_repair_before_model_evaluation"
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
    assert_execution_released(decision)
    if decision.get("protocol_status") != "frozen_for_execution":
        raise RuntimeError("Decision file is not frozen_for_execution; paid teacher calls are blocked")
    teacher = decision["teacher"]
    if teacher["model"] != TEACHER_MODEL or teacher["temperature"] != 0 or teacher["max_attempts_per_story"] != 2:
        raise RuntimeError("Teacher identity or generation contract changed")
    max_calls = teacher["pilot_max_calls"] if args.scope == "pilot" else teacher["full_max_calls"]
    cost_cap = teacher["pilot_cost_cap_usd"] if args.scope == "pilot" else teacher["full_total_cost_cap_usd"]
    from kaggle_secrets import UserSecretsClient
    if args.scope == "full":
        pilot_gate = read_json(ROOT / "pilot_gate.json", {})
        if (not pilot_gate.get("passed") or pilot_gate.get("grader_fingerprint") != grader_fingerprint()
                or pilot_gate.get("request_sha256") != digest(build()[1])
                or pilot_gate.get("decision_sha256") != digest(decision)):
            raise RuntimeError("Full generation requires a persisted passing pilot gate")
    client = BudgetedClient(UserSecretsClient().get_secret("OPENROUTER_API_KEY"), max_calls, cost_cap,
                            teacher["reserved_cost_per_call_usd"])
    stories, requests, coverage, _ = build()
    story_lookup = {s["story_id"]: s for s in stories}
    selected = [r for r in requests if args.scope == "full" or r["pilot_batch"]]
    contract_path = ROOT / ("run_contract_" + args.scope + ".json")
    contract = {"scope": args.scope, "decision_sha256": digest(decision),
              "request_sha256": digest(requests), "teacher_model": TEACHER_MODEL,
              "grader_fingerprint": grader_fingerprint(),
              "max_calls": max_calls, "cost_cap_usd": cost_cap,
              "reserved_cost_per_call_usd": teacher["reserved_cost_per_call_usd"]}
    prior_contract = read_json(contract_path)
    if prior_contract and prior_contract != contract:
        raise RuntimeError("Existing teacher contract differs")
    save_json(contract_path, contract)
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
                attempt_record = {"attempt": attempt, "raw": raw, "parsed": None, "error": None, "judgements": {}}
                try:
                    items, repair = parse_envelope(raw, pending)
                    parsed = {item["question_id"]: item["expression"].strip() for item in items}
                    attempt_record["transport_repair"] = repair
                    attempt_record["parsed"] = parsed
                    for qid in pending:
                        if qid not in parsed:
                            continue
                        judged = score("Expression: " + parsed[qid], qlookup[qid])
                        attempt_record["judgements"][qid] = judged
                        if judged["math_correct"] is True and judged["executable"]:
                            record["accepted"][qid] = {"teacher_expression": parsed[qid], "normalized_expression": judged["normalized_expression"], "judged": judged, "attempt": attempt}
                except (ValueError, json.JSONDecodeError) as exc:
                    attempt_record["error"] = type(exc).__name__ + ": " + str(exc)
                record["attempts"].append(attempt_record)
                save_json(path, record)
            print("TEACHER", position, "/", len(selected), request["story_id"], len(record["accepted"]), "/", len(qlookup), json.dumps(client.stats()), flush=True)
        verified = build_verified(stories)
        save_json(ROOT / "verified_teacher_rows.json", verified)
        report = acceptance_report(stories, args.scope)
        gate = gate_acceptance(report, teacher)
        gate.update(grader_fingerprint=grader_fingerprint(), request_sha256=digest(requests),
                    decision_sha256=digest(decision), scope=args.scope,
                    verified_rows_sha256=digest(verified))
        save_json(ROOT / (args.scope + "_acceptance.json"), report)
        save_json(ROOT / (args.scope + "_gate.json"), gate)
        save_json(ROOT / "summary.json", {"scope": args.scope, "coverage_sha256": digest(coverage),
                  "accepted": {k: len(v) for k, v in verified.items()}, "usage": client.stats(),
                  "teacher_model": TEACHER_MODEL, "reference_conditioned": False, "gate": gate})
        if not gate["passed"]:
            raise RuntimeError(args.scope + " teacher acceptance gate failed; evidence preserved")
    finally:
        save_json(ROOT / "api_usage.json", client.stats())
        package(ROOT)


if __name__ == "__main__":
    main()
