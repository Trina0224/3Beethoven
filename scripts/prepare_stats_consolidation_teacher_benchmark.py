"""Run the sealed 24-question teacher benchmark after the corpus is sealed."""
import json
from pathlib import Path

from flight_run_stats_v0_3 import read_json, save_json, package
from prepare_stats_consolidation_teacher import (BudgetedClient, DECISION, ROOT,
                                                  request_messages, parse_answers)
from stats_consolidation_grader import score, grader_fingerprint
from stats_consolidation_pilot import TEACHER_MODEL, digest, assert_execution_released, build
from stats_consolidation_semantics import validate_question

REPO = Path(__file__).resolve().parents[1]


def main():
    decision = read_json(DECISION)
    assert_execution_released(decision)
    if decision.get("protocol_status") != "frozen_for_execution":
        raise RuntimeError("Protocol is not frozen")
    full_gate = read_json(ROOT / "full_gate.json", {})
    if (not full_gate.get("passed") or full_gate.get("grader_fingerprint") != grader_fingerprint()
            or full_gate.get("request_sha256") != digest(build()[1])
            or full_gate.get("decision_sha256") != digest(decision)):
        raise RuntimeError("Teacher corpus must be sealed behind a passing full gate")
    questions = read_json(REPO / "docs/STATS_CONSOLIDATION_TEACHER_BENCHMARK.json")
    if len(questions) != 24:
        raise RuntimeError("Teacher benchmark is not the frozen 24-question subset")
    for q in questions:
        validate_question(q)
    from kaggle_secrets import UserSecretsClient
    rule = decision["teacher"]
    client = BudgetedClient(UserSecretsClient().get_secret("OPENROUTER_API_KEY"),
                            rule["total_max_calls_including_benchmark"],
                            rule["total_cost_cap_usd_including_benchmark"],
                            rule["reserved_cost_per_call_usd"])
    contract = {"teacher_model": TEACHER_MODEL, "decision_sha256": digest(decision),
                "grader_fingerprint": grader_fingerprint(),
                "benchmark_sha256": digest(questions), "corpus_sha256": digest(read_json(ROOT / "verified_teacher_rows.json"))}
    prior = read_json(ROOT / "teacher_benchmark_contract.json")
    if prior and prior != contract:
        raise RuntimeError("Teacher benchmark contract drift")
    save_json(ROOT / "teacher_benchmark_contract.json", contract)
    rows = read_json(ROOT / "teacher_benchmark_results.json", [])
    done = {r["id"] for r in rows}
    for q in questions:
        if q["id"] in done:
            continue
        request = {"story_id": q["id"], "questions": [{"question_id": q["id"], "question": q["question"]}]}
        raw = client.call("sealed_benchmark_" + q["id"], request_messages(request, {q["id"]}),
                          max_tokens=160, json_mode=True)
        parsed = parse_answers(raw)
        expression = parsed.get(q["id"], "")
        judged = score("Expression: " + expression, q)
        rows.append({"id": q["id"], "category": q["category"], "raw": raw,
                     "expression": expression, **judged})
        save_json(ROOT / "teacher_benchmark_results.json", rows)
    save_json(ROOT / "teacher_benchmark_summary.json",
              {"n": len(rows), "correct": sum(r["primary_correct"] for r in rows),
               "pending": sum(r["review_required"] for r in rows), "usage": client.stats()})
    package(ROOT)


if __name__ == "__main__":
    main()
