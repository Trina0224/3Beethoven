"""3Beethoven teacher-data pilot v0.3

Curriculum-driven generation. The 70B model no longer chooses the curriculum.
For each explicit concept, generate one educational item, critique it, optionally repair it,
and accept only validated records.

Pipeline:
  fixed curriculum -> 70B generator -> 70B critic -> PASS / REPAIR / REJECT

This is a small harmony/counterpoint pilot only. Do not scale until manually reviewed.
"""

import json
import re
import time
from pathlib import Path
import requests
from kaggle_secrets import UserSecretsClient

MODEL = "meta-llama/llama-3.3-70b-instruct"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
OUT = Path("/kaggle/working/teacher_pilot_v0_3.jsonl")
REJECTS = Path("/kaggle/working/teacher_pilot_v0_3_rejects.jsonl")

# Explicit syllabus: one item per concept. No free-topic generation.
CURRICULUM = [
    {"concept": "secondary_dominant", "task": "explain_function_and_resolution"},
    {"concept": "leading_tone_seventh_chord", "task": "identify_tendency_tones_and_resolution"},
    {"concept": "neapolitan_sixth", "task": "explain_spelling_function_and_typical_resolution"},
    {"concept": "italian_augmented_sixth", "task": "identify_pitches_and_voice_leading"},
    {"concept": "french_augmented_sixth", "task": "distinguish_from_italian_augmented_sixth"},
    {"concept": "german_augmented_sixth", "task": "explain_resolution_and_parallel_fifths_issue"},
    {"concept": "cadential_six_four", "task": "explain_function_and_voice_leading"},
    {"concept": "deceptive_cadence", "task": "explain_function_and_common_voice_leading"},
    {"concept": "suspension", "task": "define_preparation_dissonance_resolution"},
    {"concept": "retardation", "task": "distinguish_from_suspension"},
    {"concept": "appoggiatura", "task": "distinguish_from_suspension_and_passing_tone"},
    {"concept": "passing_tone", "task": "define_and_distinguish_from_neighbor_tone"},
    {"concept": "neighbor_tone", "task": "define_and_distinguish_from_passing_tone"},
    {"concept": "oblique_motion", "task": "define_and_compare_with_contrary_motion"},
    {"concept": "invertible_counterpoint_at_octave", "task": "explain_voice_exchange_and_interval_consequences"},
]

SYSTEM_GENERATOR = """You are a careful undergraduate-level music-theory instructor.
Generate exactly ONE synthetic training example for the concept and task specified by the user.
Do not invent a different topic. Accuracy and standard terminology matter more than style.
If conventions vary by school or repertory, qualify the statement.
Stay within Western classical common-practice harmony/counterpoint unless the concept itself requires otherwise.

Return VALID JSON ONLY with exactly these string fields:
concept, task, question, answer, explanation, common_mistake.
No markdown. No extra keys.
"""

SYSTEM_CRITIC = """You are a strict conservatory-level music-theory reviewer.
Review the candidate for factual correctness, standard terminology, ambiguity,
voice-leading accuracy, and whether the explanation supports the answer.

Return VALID JSON ONLY:
{
  "verdict": "PASS" | "REPAIR" | "REJECT",
  "reason": "short precise reason",
  "repaired": {
    "concept": "...",
    "task": "...",
    "question": "...",
    "answer": "...",
    "explanation": "...",
    "common_mistake": "..."
  }
}

Rules:
- PASS only if teachable as written.
- REPAIR if the core item is sound but one or more statements need a small factual/terminological correction.
- REJECT if the premise is wrong, ambiguous, off-topic, or requires major rewriting.
- For PASS or REJECT, set repaired fields to empty strings.
- Never soften a factual error just to pass it.
"""


def call_llm(api_key, system, user, temperature=0.2, max_tokens=900):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(API_URL, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def parse_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def valid_item(obj, spec):
    fields = ["concept", "task", "question", "answer", "explanation", "common_mistake"]
    if not isinstance(obj, dict):
        return False, "not object"
    if any(not isinstance(obj.get(k), str) or not obj[k].strip() for k in fields):
        return False, "missing required field"
    if obj["concept"] != spec["concept"]:
        return False, f"wrong concept {obj['concept']}"
    if obj["task"] != spec["task"]:
        return False, f"wrong task {obj['task']}"
    if len(obj["explanation"]) < 80:
        return False, "explanation too short"
    return True, ""


def generate(api_key, spec):
    user = f"""Concept: {spec['concept']}
Task: {spec['task']}

Create one precise educational item for this exact concept and task.
Prefer concrete theory/voice-leading facts over vague listener psychology.
"""
    return parse_json(call_llm(api_key, SYSTEM_GENERATOR, user, temperature=0.15, max_tokens=850))


def critique(api_key, candidate):
    user = "Review this candidate:\n" + json.dumps(candidate, ensure_ascii=False, indent=2)
    return parse_json(call_llm(api_key, SYSTEM_CRITIC, user, temperature=0.0, max_tokens=700))


def write_jsonl(path, obj):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    api_key = UserSecretsClient().get_secret("OPENROUTER_API_KEY")
    OUT.unlink(missing_ok=True)
    REJECTS.unlink(missing_ok=True)

    accepted = []
    stats = {"PASS": 0, "REPAIR": 0, "REJECT": 0, "ERROR": 0}
    api_calls = 0

    for i, spec in enumerate(CURRICULUM, 1):
        print(f"\n[{i:02d}/{len(CURRICULUM)}] {spec['concept']} / {spec['task']}")

        candidate = None
        # At most 2 generator attempts for malformed/invalid output.
        for gen_try in range(1, 3):
            try:
                candidate = generate(api_key, spec)
                api_calls += 1
                ok, reason = valid_item(candidate, spec)
                if not ok:
                    raise ValueError(reason)
                break
            except Exception as e:
                print(f"  generator try {gen_try}: {e}")
                candidate = None
                time.sleep(0.7)

        if candidate is None:
            stats["ERROR"] += 1
            write_jsonl(REJECTS, {"spec": spec, "stage": "generator", "reason": "failed after 2 tries"})
            continue

        try:
            review = critique(api_key, candidate)
            api_calls += 1
        except Exception as e:
            stats["ERROR"] += 1
            print("  critic error:", e)
            write_jsonl(REJECTS, {"spec": spec, "stage": "critic", "reason": str(e), "candidate": candidate})
            continue

        verdict = review.get("verdict")

        if verdict == "PASS":
            final = candidate
            stats["PASS"] += 1
            print("  PASS -", review.get("reason", "")[:120])

        elif verdict == "REPAIR":
            repaired = review.get("repaired", {})
            ok, reason = valid_item(repaired, spec)
            if not ok:
                stats["REJECT"] += 1
                print("  REJECT bad repair -", reason)
                write_jsonl(REJECTS, {"spec": spec, "stage": "repair", "reason": reason, "candidate": candidate, "review": review})
                continue

            # One second validation round only.
            try:
                review2 = critique(api_key, repaired)
                api_calls += 1
            except Exception as e:
                stats["ERROR"] += 1
                print("  repair validation error:", e)
                write_jsonl(REJECTS, {"spec": spec, "stage": "repair_validation", "reason": str(e), "candidate": repaired})
                continue

            if review2.get("verdict") != "PASS":
                stats["REJECT"] += 1
                print("  REJECT repair validation -", review2.get("reason", "")[:120])
                write_jsonl(REJECTS, {"spec": spec, "stage": "repair_validation", "candidate": repaired, "review": review2})
                continue

            final = repaired
            stats["REPAIR"] += 1
            print("  REPAIR->PASS -", review2.get("reason", "")[:120])

        else:
            stats["REJECT"] += 1
            print("  REJECT -", review.get("reason", "")[:120])
            write_jsonl(REJECTS, {"spec": spec, "stage": "critic", "candidate": candidate, "review": review})
            continue

        record = {
            **final,
            "teacher_model": MODEL,
            "critic_model": MODEL,
            "pipeline_version": "v0.3",
        }
        accepted.append(record)
        write_jsonl(OUT, record)

    print("\n==============================")
    print("TEACHER PILOT v0.3 COMPLETE")
    print("==============================")
    print("Accepted:", len(accepted), "/", len(CURRICULUM))
    print("Stats:", stats)
    print("Approx. API calls:", api_calls)
    print("Accepted file:", OUT)
    print("Reject log:", REJECTS)

    print("\n--- ACCEPTED SAMPLES ---")
    for i, x in enumerate(accepted[:8], 1):
        print(f"\n[{i}] {x['concept']}")
        print("Q:", x["question"])
        print("A:", x["answer"])
        print("Why:", x["explanation"])
        print("Common mistake:", x["common_mistake"])


if __name__ == "__main__":
    main()
