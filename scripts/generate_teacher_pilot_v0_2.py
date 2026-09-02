"""3Beethoven teacher-data pilot v0.2.

Pipeline:
    Llama 3.3 70B generator -> independent critic prompt -> filter -> JSONL

Goals:
- Retry malformed JSON instead of silently losing examples.
- Reject ambiguous, subjective, off-domain, or factually weak examples.
- Reject near-duplicate questions within the pilot batch.
- Keep the frozen benchmark out of training data.

Expected environment:
- Kaggle notebook
- OPENROUTER_API_KEY stored in Kaggle Secrets
- requests installed

No secrets are written to output files.
"""

import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

import requests
from kaggle_secrets import UserSecretsClient

MODEL = "meta-llama/llama-3.3-70b-instruct"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
OUT = Path("/kaggle/working/teacher_pilot_v0_2.jsonl")
REJECTS = Path("/kaggle/working/teacher_pilot_v0_2_rejects.jsonl")

# Small pilot only. Do not scale until this batch is manually reviewed.
TARGETS = {
    "harmony_counterpoint": 20,
    "orchestration": 8,
    "form_analysis": 6,
    "history_context": 4,
    "style_comparison": 2,
}

CATEGORY_GUIDANCE = {
    "harmony_counterpoint": """
Focus on common-practice harmony and academically standard counterpoint concepts:
voice leading, tendency tones, suspensions/retardations/appoggiaturas, non-chord tones,
secondary dominants, cadences, chromatic predominant chords, species counterpoint,
invertible counterpoint, imitation, fugue technique, and dissonance treatment.
Avoid vague psychology of listening and avoid claims that depend mainly on taste.
""",
    "orchestration": """
Focus on objectively teachable orchestration and notation:
transposing instruments, ranges/registers, clefs, doublings, divisi, articulations,
string techniques, brass/woodwind families, balance, and conventional notation.
Prefer questions with a clear technical answer over subjective claims about beauty or mood.
""",
    "form_analysis": """
Focus on academically standard formal functions and structures:
sonata form, binary/rounded binary, rondo, ritornello, fugue, variation forms,
da capo aria, passacaglia/chaconne, cyclic form, and thematic transformation.
Use precise terminology and avoid pretending every repertoire example follows one rigid template.
""",
    "history_context": """
Focus on well-established historical facts and relationships in Western classical music:
patronage, institutions, premieres, careers, public concert culture, historical movements,
and documented composer/work context. Avoid disputed anecdotes unless the uncertainty is explicit.
""",
    "style_comparison": """
Focus on well-established stylistic distinctions supported by musicology:
period idioms, compositional techniques, harmonic language, formal practice, and orchestration.
Do not ask which composer is 'better', 'more emotional', or otherwise subjective.
""",
}

SYSTEM_GENERATOR = """You are generating high-quality synthetic training data for a small
classical-music specialist language model. Accuracy matters more than cleverness.
You are NOT writing trivia filler. Produce one educational example whose answer would be
accepted in a serious undergraduate music-theory/history/orchestration course.

Return exactly one JSON object with these string fields:
category, question, answer, explanation, common_mistake.

Rules:
1. The answer must be factually defensible and precise.
2. If terminology varies by school or period, explicitly qualify the answer instead of
   presenting a debatable convention as universal.
3. Do not create subjective aesthetics questions.
4. Do not create jazz/pop questions; stay within Western classical music for this project.
5. Do not ask trivial composer-date facts unless historical context is genuinely instructive.
6. Avoid repeating common templates such as fugue-vs-canon or suspension-vs-appoggiatura
   unless the requested category specifically needs a new, materially different concept.
7. Never mention this dataset, the benchmark, Llama, distillation, or the student model.
8. Output valid JSON only. No markdown fences.
"""

SYSTEM_CRITIC = """You are a strict conservatory-level reviewer of synthetic classical-music
training data. Your job is to REJECT weak examples. Do not be polite to the generator.

Check the candidate for:
- factual correctness
- precise terminology
- ambiguity or overgeneralization
- whether the explanation actually supports the answer
- whether the common_mistake is itself accurate
- whether the topic belongs to Western classical music
- whether the item is academically useful rather than subjective or filler

Return exactly one JSON object:
{
  "verdict": "PASS" or "REJECT",
  "reason": "short reason",
  "corrected_answer": "" or a corrected answer if the candidate is repairable
}

PASS only if you would be comfortable teaching the candidate as written.
If any substantive factual correction is needed, REJECT it rather than silently repairing it.
Output valid JSON only. No markdown fences.
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


def parse_json_object(text):
    """Parse strict JSON, with a conservative fallback extracting the outer object."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_question(q):
    q = q.lower()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def too_similar(question, accepted_questions, threshold=0.84):
    q = normalize_question(question)
    for prior in accepted_questions:
        ratio = SequenceMatcher(None, q, normalize_question(prior)).ratio()
        if ratio >= threshold:
            return True, ratio, prior
    return False, 0.0, None


def valid_candidate(obj, category):
    required = ["category", "question", "answer", "explanation", "common_mistake"]
    if not isinstance(obj, dict):
        return False, "not an object"
    if any(not isinstance(obj.get(k), str) or not obj.get(k).strip() for k in required):
        return False, "missing/empty required field"
    if obj["category"] != category:
        return False, f"wrong category: {obj['category']}"
    if len(obj["question"]) < 25:
        return False, "question too short"
    if len(obj["explanation"]) < 80:
        return False, "explanation too short"
    return True, ""


def generate_candidate(api_key, category, attempt_context):
    user = f"""Category: {category}

Category guidance:
{CATEGORY_GUIDANCE[category]}

Create ONE new example. It must be materially different from examples already accepted in this run.
Variation hint for this attempt: {attempt_context}
"""
    raw = call_llm(api_key, SYSTEM_GENERATOR, user, temperature=0.45, max_tokens=1000)
    return parse_json_object(raw)


def critique_candidate(api_key, candidate):
    user = "Review this candidate:\n" + json.dumps(candidate, ensure_ascii=False, indent=2)
    raw = call_llm(api_key, SYSTEM_CRITIC, user, temperature=0.0, max_tokens=450)
    return parse_json_object(raw)


def write_jsonl(path, obj):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    secrets = UserSecretsClient()
    api_key = secrets.get_secret("OPENROUTER_API_KEY")

    OUT.unlink(missing_ok=True)
    REJECTS.unlink(missing_ok=True)

    accepted = []
    accepted_questions = []
    total_api_calls = 0

    for category, target in TARGETS.items():
        print(f"\nTarget: {target} PASS examples for {category}")
        category_pass = 0
        attempts = 0
        max_attempts = target * 5

        while category_pass < target and attempts < max_attempts:
            attempts += 1
            attempt_context = (
                f"attempt {attempts}; choose a concept not already overrepresented; "
                "prefer a concrete distinction, analytical principle, or technically verifiable fact"
            )

            # ---------- Generator with JSON retry ----------
            candidate = None
            last_error = None
            for gen_try in range(1, 4):
                try:
                    candidate = generate_candidate(api_key, category, attempt_context)
                    total_api_calls += 1
                    ok, reason = valid_candidate(candidate, category)
                    if not ok:
                        raise ValueError(reason)
                    break
                except Exception as e:
                    last_error = str(e)
                    if "total_api_calls" not in locals():
                        pass
                    time.sleep(0.8 * gen_try)

            if candidate is None:
                print(f"  attempt {attempts:02d}: GENERATOR FAILED - {last_error}")
                write_jsonl(REJECTS, {
                    "category": category,
                    "stage": "generator",
                    "reason": last_error,
                })
                continue

            # ---------- Duplicate filter before paying for critic ----------
            duplicate, ratio, prior = too_similar(candidate["question"], accepted_questions)
            if duplicate:
                print(f"  attempt {attempts:02d}: REJECT duplicate ({ratio:.2f})")
                write_jsonl(REJECTS, {
                    "category": category,
                    "stage": "dedupe",
                    "reason": f"similarity={ratio:.3f}",
                    "question": candidate["question"],
                    "similar_to": prior,
                })
                continue

            # ---------- Critic with JSON retry ----------
            review = None
            last_error = None
            for critic_try in range(1, 4):
                try:
                    review = critique_candidate(api_key, candidate)
                    total_api_calls += 1
                    if review.get("verdict") not in {"PASS", "REJECT"}:
                        raise ValueError("critic returned invalid verdict")
                    break
                except Exception as e:
                    last_error = str(e)
                    time.sleep(0.8 * critic_try)

            if review is None:
                print(f"  attempt {attempts:02d}: CRITIC FAILED - {last_error}")
                write_jsonl(REJECTS, {
                    "category": category,
                    "stage": "critic",
                    "reason": last_error,
                    "candidate": candidate,
                })
                continue

            if review["verdict"] != "PASS":
                print(f"  attempt {attempts:02d}: REJECT critic - {review.get('reason', '')[:90]}")
                write_jsonl(REJECTS, {
                    "category": category,
                    "stage": "critic",
                    "review": review,
                    "candidate": candidate,
                })
                continue

            # ---------- Accept ----------
            record = {
                **candidate,
                "teacher_model": MODEL,
                "critic_model": MODEL,
                "critic_reason": review.get("reason", ""),
                "pipeline_version": "v0.2",
            }
            accepted.append(record)
            accepted_questions.append(candidate["question"])
            write_jsonl(OUT, record)
            category_pass += 1

            print(
                f"  PASS {category_pass:02d}/{target} "
                f"(attempt {attempts:02d}) - {candidate['question'][:72]}"
            )

        if category_pass < target:
            print(f"  WARNING: only {category_pass}/{target} passed after {attempts} attempts")

    print("\n==============================")
    print("TEACHER PILOT v0.2 COMPLETE")
    print("==============================")
    print(f"Accepted: {len(accepted)} / {sum(TARGETS.values())}")
    print(f"Approx. API calls: {total_api_calls}")
    print(f"Accepted file: {OUT}")
    print(f"Reject log:    {REJECTS}")

    print("\n--- FIRST 5 ACCEPTED SAMPLES ---")
    for i, x in enumerate(accepted[:5], 1):
        print(f"\n[{i}] {x['category']}")
        print("Q:", x["question"])
        print("A:", x["answer"])
        print("Why:", x["explanation"])
        print("Common mistake:", x["common_mistake"])
        print("Critic:", x["critic_reason"])


if __name__ == "__main__":
    main()
