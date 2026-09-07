"""Frozen, CPU-only, paired teacher development screen. No student training."""
import argparse
import json
import math
import random
from collections import Counter
from fractions import Fraction as F
from pathlib import Path

from exact_calculator import calculate
from flight_run_stats_v0_3 import save_json, read_json, read_jsonl, append, package
from stats_consolidation_pilot import DOCS, bundle, digest, historical_replay_questions
from stats_consolidation_semantics import task_key, spec_for, validate_question
from stats_consolidation_grader import score, outcome, grader_fingerprint
from stats_teacher_envelope import parse_answers, _unique_object

MODEL = "meta-llama/llama-3.3-70b-instruct"
ROOT = Path("/kaggle/working/3beethoven_teacher_ab")
DATA = DOCS / "STATS_TEACHER_AB_DATA.json"
PRIOR_CALLS, PRIOR_COST = 38, 0.00402078


def all_questions(value):
    if isinstance(value, dict):
        if {"question", "expression", "category", "id"} <= value.keys():
            yield value
        else:
            for v in value.values():
                yield from all_questions(v)
    elif isinstance(value, list):
        for v in value:
            yield from all_questions(v)


def build_development():
    # This offline collision audit never supplies held-out questions to the API.
    occupied = {task_key(q) for q in historical_replay_questions()}
    for name in ("CANDIDATE_STORIES", "SELECTION_VALIDATION", "HOLDOUT"):
        for q in all_questions(read_json(DOCS / ("STATS_CONSOLIDATION_"+name+".json"))):
            occupied.add(task_key(q))
    rng, rows = random.Random(210907), []
    for family, index in (("poisson_process", 0), ("interval", 0), ("affine_poisson", 2)):
        for i in range(8):
            for attempt in range(1000):
                q = bundle(family, "development", i*1000+attempt, rng)["questions"][index]
                if task_key(q) not in occupied:
                    break
            else:
                raise ValueError("Could not find disjoint development question")
            occupied.add(task_key(q))
            q["id"] = f"teacher_ab_{family}_{i:02d}"
            q["development_family"] = family
            q["provenance"] = "development_only; not training or final evaluation"
            validate_question(q)
            rows.append(q)
    return {"seed": 210907, "role": "development_only", "questions": rows}


def intermediates(q):
    s = spec_for(q)
    if s["kind"] == "process":
        return {"duration_minutes": F(s["duration"]), "count_mean": F(s["rate"])*F(s["duration"])}
    if s["kind"] == "interval":
        lo, hi, d = F(s["lower"]), F(s["upper"]), F(s["divisor"])
        return {"center": (lo+hi)/2, "old_half_width": (hi-lo)/2, "new_half_width": (hi-lo)/(2*d)}
    m, a, b = F(s["mean"]), F(s["scale"]), F(s["offset"])
    return {"mean_Y": a*m+b, "variance_Y": a*a*m}


def messages(q, arm):
    prompt = {"question_id": q["id"], "question": q["question"]}
    instruction = ("Solve this statistics question independently. Return a JSON object with answers: "
                   "a list containing one object with string question_id and expression. "
                   "The final expression must be fully numerical using integers, +,-,*,/,**,comb(n,r). "
                   "Keep arithmetic, unit conversions and powers unevaluated. No variables or prose.")
    if arm == "B":
        instruction += (" Before answers include an intermediates object. Its required keys are "
                        + ", ".join(intermediates(q)) + ". Each value must be a numerical expression string. "
                        "Calculate these quantities first, then construct the final answer.")
    return [{"role": "system", "content": instruction},
            {"role": "user", "content": json.dumps(prompt, separators=(",", ":"))}]


def judge(raw, q, arm):
    result = {"accepted": False, "classification": "malformed_response"}
    try:
        items, repair = parse_answers(raw, {q["id"]})
        result.update(expression=items[0]["expression"], transport_repair=repair)
        judged = score("Expression: " + items[0]["expression"], q)
        result.update(judged=judged, classification=outcome(judged))
        steps_ok = True
        if arm == "B":
            # B has one strict top-level intermediate object; no speculative recovery.
            obj = json.loads(raw, object_pairs_hook=_unique_object)
            steps = obj["intermediates"]
            expected = intermediates(q)
            if not isinstance(steps, dict) or set(steps) != set(expected):
                raise ValueError("Incomplete intermediate quantities")
            checks = {k: isinstance(steps[k], str) and F(calculate(steps[k])) == v for k,v in expected.items()}
            steps_ok = all(checks.values())
            result.update(intermediate_checks=checks, intermediate_raw=steps)
        result["accepted"] = result["classification"] == "accepted" and steps_ok
        if not steps_ok:
            result["classification"] = "intermediate_error"
    except (ValueError, KeyError, TypeError, SyntaxError, ZeroDivisionError) as exc:
        result.update(accepted=False, classification="malformed_response", error=str(exc))
    return result


def call(root, key, tag, msgs):
    payload = {"model": MODEL, "messages": msgs, "temperature": 0, "max_tokens": 400,
               "response_format": {"type": "json_object"},
               "provider": {"only": ["deepinfra"], "allow_fallbacks": False, "require_parameters": True,
                            "enforce_distillable_text": True,
                            "max_price": {"prompt": 1, "completion": 2, "request": 0}}}
    signature = digest(payload)
    path, ledger = root / "api_cache" / (tag+".json"), root / "api_ledger.jsonl"
    cached = read_json(path)
    if cached:
        if cached["request_sha256"] != signature:
            raise RuntimeError("Changed cached request")
        return cached
    history = read_jsonl(ledger)
    starts = {r["tag"] for r in history if r["event"] == "started"}
    responses = [r for r in history if r["event"] == "response"]
    if starts - {r["tag"] for r in responses}:
        raise RuntimeError("Unresolved API attempt; no automatic retry")
    costs = [r["usage"].get("cost") for r in responses]
    if any(c is None or not isinstance(c, (int,float)) or not math.isfinite(c) or c < 0 for c in costs):
        raise RuntimeError("Unknown/invalid cost; stopped")
    if len(starts) >= 48 or len(starts)+PRIOR_CALLS >= 216 or sum(costs)+PRIOR_COST+0.01 > 0.50:
        raise RuntimeError("Frozen spending/call gate reached")
    if sum(len(m["content"].encode()) for m in msgs) > 4000:
        raise RuntimeError("Request byte bound exceeded")
    import requests
    append(ledger, {"event": "started", "tag": tag, "request_sha256": signature})
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                 headers={"Authorization": "Bearer " + key}, json=payload, timeout=90)
    except requests.RequestException:
        raise RuntimeError("Network failure; no automatic retry") from None
    if response.status_code != 200:
        append(ledger, {"event": "http_error", "tag": tag, "status": response.status_code})
        raise RuntimeError(f"HTTP {response.status_code}; body omitted; no automatic retry")
    data = response.json()
    choice = data.get("choices", [{}])[0]
    record = {"request_sha256": signature, "request": payload, "text": choice.get("message", {}).get("content"),
              "model": data.get("model"), "provider": data.get("provider"), "id": data.get("id"),
              "finish_reason": choice.get("finish_reason"), "usage": data.get("usage", {})}
    save_json(path, record)
    append(ledger, {"event": "response", "tag": tag, "usage": record["usage"]})
    return record


def summarize(rows, completed):
    arms = {}
    for arm in ("A", "B"):
        selected = [r for r in rows if r["arm"] == arm]
        counts = Counter(r["family"] for r in selected if r["accepted"])
        passed = completed and sum(counts.values()) >= 22 and all(counts[f] >= 7 for f in ("poisson_process", "interval", "affine_poisson"))
        pending = any(r["classification"] == "equivalence_pending" for r in selected)
        arms[arm] = {"accepted": sum(counts.values()), "n": len(selected), "by_family": dict(counts), "passed": passed and not pending}
    eligible = [a for a in arms if arms[a]["passed"]]
    winner = max(eligible, key=lambda a: (arms[a]["accepted"], a == "A")) if eligible else None
    paired = {}
    for r in rows:
        paired.setdefault(r["id"], {})[r["arm"]] = r["accepted"]
    flips = Counter(f"A{int(p['A'])}_B{int(p['B'])}" for p in paired.values() if set(p) == {"A", "B"})
    return {"role": "development_only", "completed": completed, "arms": arms,
            "selected_arm": winner, "paired": dict(flips), "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("build", "validate", "run"), default="validate")
    args = parser.parse_args()
    if args.stage == "build":
        if DATA.exists():
            raise SystemExit("Development data already frozen")
        save_json(DATA, build_development())
        print("FROZEN", digest(read_json(DATA)))
        return
    data = read_json(DATA)
    if len(data["questions"]) != 24:
        raise ValueError("Expected 24 development questions")
    for q in data["questions"]:
        validate_question(q)
        for arm in ("A", "B"):
            value = {"answers": [{"question_id": q["id"], "expression": q["expression"]}]}
            if arm == "B":
                value["intermediates"] = {k:str(v) for k,v in intermediates(q).items()}
            assert judge(json.dumps(value), q, arm)["accepted"], q["id"]
    if args.stage == "validate":
        print("VALIDATED 24 questions, 48 reference fixtures", digest(data))
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {"data_sha256": digest(data), "grader_fingerprint": grader_fingerprint(),
                "runner_sha256": digest(Path(__file__).read_text()), "prior_calls": PRIOR_CALLS, "prior_cost": PRIOR_COST}
    existing = read_json(ROOT / "run_manifest.json")
    if existing and existing != manifest:
        raise RuntimeError("Run manifest mismatch")
    save_json(ROOT / "run_manifest.json", manifest)
    from kaggle_secrets import UserSecretsClient
    key = UserSecretsClient().get_secret("OPENROUTER_API_KEY")
    rows, completed = [], False
    try:
        for i,q in enumerate(data["questions"]):
            for arm in (("A", "B") if i%2 == 0 else ("B", "A")):
                record = call(ROOT, key, q["id"]+"_"+arm, messages(q, arm))
                if record["model"] != MODEL or record["provider"] != "DeepInfra":
                    raise RuntimeError("Model/provider pin mismatch")
                if record["usage"].get("cost") is None:
                    raise RuntimeError("Missing cost; stopped")
                verdict = judge(record["text"], q, arm) if isinstance(record["text"], str) else {"accepted": False, "classification": "non_text"}
                if record["finish_reason"] != "stop":
                    verdict.update(accepted=False, classification="non_stop_finish")
                row = {"id": q["id"], "arm": arm, "family": q["development_family"], "raw": record["text"], **verdict}
                rows.append(row)
                save_json(ROOT / "results.json", summarize(rows, False))
                print("AB", len(rows), "/48", q["id"], arm, verdict["classification"], flush=True)
        completed = True
    except Exception as exc:
        save_json(ROOT / "stop.json", {"error": str(exc), "type": type(exc).__name__, "completed_responses": len(rows)})
        raise
    finally:
        result = summarize(rows, completed)
        save_json(ROOT / "results.json", result)
        history = read_jsonl(ROOT / "api_ledger.jsonl")
        save_json(ROOT / "usage.json", {"prior_calls": PRIOR_CALLS, "prior_cost": PRIOR_COST,
                  "calls": sum(r["event"] == "started" for r in history),
                  "new_cost": sum(r["usage"].get("cost", 0) or 0 for r in history if r["event"] == "response")})
        package(ROOT)
        print("AB_SUMMARY", json.dumps({k:v for k,v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
