"""3Beethoven statistics teacher-data pilot v0.1.

Goal:
- Generate a small, high-quality synthetic dataset only in domains where the 70B teacher
  demonstrated a large and reliable advantage over the 3B student.
- Current scope: inference/hypothesis testing and distributions/expectation.
- Do NOT generate Bayes/probability training data in this version.
- Do NOT generate regression/causality training data in this version because the 3B baseline
  already matched the 70B teacher on the frozen diagnostic.

Pipeline:
    fixed curriculum concept -> Llama 3.3 70B generator -> deterministic validation -> JSONL

This is a pilot. Manually inspect before scaling.
"""

import json
import math
import re
import time
from pathlib import Path

import requests
from kaggle_secrets import UserSecretsClient

MODEL = "meta-llama/llama-3.3-70b-instruct"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
OUT = Path("/kaggle/working/stats_teacher_pilot_v0_1.jsonl")
REJECTS = Path("/kaggle/working/stats_teacher_pilot_v0_1_rejects.jsonl")

# 24 concepts total: 14 inference/testing + 10 distributions/expectation.
# One example per concept in this pilot to maximize coverage and minimize duplication.
CURRICULUM = [
    # inference / hypothesis testing
    ("inference_testing", "p_value_interpretation", "interpret a p-value correctly without treating it as P(H0 is true)"),
    ("inference_testing", "type_i_error", "distinguish Type I error from Type II error"),
    ("inference_testing", "type_ii_error", "distinguish Type II error from Type I error"),
    ("inference_testing", "statistical_power", "interpret power as P(reject H0 | H0 is false)"),
    ("inference_testing", "alpha_tradeoff", "explain how changing alpha changes Type I error risk and, all else equal, power"),
    ("inference_testing", "confidence_interval_width", "reason about sample size and confidence level effects on CI width"),
    ("inference_testing", "ci_and_two_sided_test", "connect a two-sided alpha=.05 test to whether a 95% CI excludes the null value"),
    ("inference_testing", "independent_t_test", "select an independent-samples t-test for two independent group means under standard assumptions"),
    ("inference_testing", "paired_t_test", "select a paired t-test for before/after or matched observations"),
    ("inference_testing", "chi_square_independence", "select a chi-square test of independence for two categorical variables"),
    ("inference_testing", "one_sample_mean_test", "recognize a one-sample mean test against a fixed reference value"),
    ("inference_testing", "multiple_testing", "explain why repeated hypothesis tests inflate family-wise false-positive risk"),
    ("inference_testing", "practical_vs_statistical_significance", "distinguish statistical significance from practical importance"),
    ("inference_testing", "sample_size_power", "explain why larger sample size generally increases power when effect size and alpha are fixed"),

    # distributions / expectation
    ("distributions_expectation", "bernoulli_mean_variance", "use E[X]=p and Var(X)=p(1-p) for Bernoulli X"),
    ("distributions_expectation", "binomial_mean_variance", "use E[X]=np and Var(X)=np(1-p) for Binomial X"),
    ("distributions_expectation", "poisson_mean_variance", "use mean=lambda and variance=lambda for Poisson X"),
    ("distributions_expectation", "linear_expectation", "apply E[aX+b]=aE[X]+b"),
    ("distributions_expectation", "variance_scaling", "apply Var(aX+b)=a^2 Var(X)"),
    ("distributions_expectation", "variance_sum_independent", "use Var(X+Y)=Var(X)+Var(Y) for independent X,Y"),
    ("distributions_expectation", "normal_68_95_997", "apply the empirical 68-95-99.7 rule for normal data"),
    ("distributions_expectation", "uniform_expectation", "compute the mean of Uniform(a,b) as (a+b)/2"),
    ("distributions_expectation", "exponential_memoryless", "recognize the exponential distribution's memoryless property"),
    ("distributions_expectation", "poisson_modeling", "recognize when a Poisson count model is appropriate for event counts over a fixed interval"),
]

SYSTEM = """You are generating synthetic training examples for a small local language model
that is being taught introductory-to-intermediate statistical reasoning.

Accuracy and pedagogical clarity matter more than cleverness.

Return exactly one valid JSON object with these string fields:
category, concept, question, answer, explanation, common_mistake.

Rules:
1. Stay exactly on the requested concept. Do not broaden into unrelated statistics topics.
2. The question must be NEW and must not copy benchmark wording.
3. Prefer a concrete scenario that requires understanding, not trivia.
4. The answer must be short but complete.
5. The explanation must explicitly state the decisive statistical principle.
6. The common_mistake must describe a realistic misconception.
7. Do not ask for unavailable software or external data.
8. Do not use Bayes theorem or advanced Bayesian statistics in this pilot.
9. Do not use regression or causal-inference topics in this pilot.
10. Output JSON only. No markdown fences.
"""


def call_llm(api_key, user, temperature=0.35, max_tokens=900):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(API_URL, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def parse_json_object(text):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def validate(obj, category, concept):
    required = ["category", "concept", "question", "answer", "explanation", "common_mistake"]
    if not isinstance(obj, dict):
        return False, "not an object"
    for key in required:
        if not isinstance(obj.get(key), str) or not obj[key].strip():
            return False, f"missing/empty {key}"
    if obj["category"] != category:
        return False, f"wrong category: {obj['category']}"
    if obj["concept"] != concept:
        return False, f"wrong concept: {obj['concept']}"
    if len(obj["question"]) < 30:
        return False, "question too short"
    if len(obj["explanation"]) < 100:
        return False, "explanation too short"
    return True, ""


def write_jsonl(path, obj):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    secrets = UserSecretsClient()
    api_key = secrets.get_secret("OPENROUTER_API_KEY")

    OUT.unlink(missing_ok=True)
    REJECTS.unlink(missing_ok=True)

    accepted = []
    calls = 0

    print(f"Generating {len(CURRICULUM)} statistics teacher pilot examples...\n")

    for idx, (category, concept, target) in enumerate(CURRICULUM, 1):
        user = f"""Category: {category}
Concept ID: {concept}
Target skill: {target}

Create exactly one training example for this concept.
Use a scenario and numbers/wording different from common textbook one-liners where possible.
"""

        record = None
        last_error = None

        # Up to two tries only. This is a pilot, not an unlimited retry loop.
        for attempt in range(1, 3):
            try:
                raw = call_llm(api_key, user)
                calls += 1
                obj = parse_json_object(raw)
                ok, reason = validate(obj, category, concept)
                if not ok:
                    raise ValueError(reason)
                record = {
                    **obj,
                    "teacher_model": MODEL,
                    "pipeline_version": "stats_v0.1",
                }
                break
            except Exception as e:
                last_error = str(e)
                time.sleep(0.7 * attempt)

        if record is None:
            print(f"[{idx:02d}/{len(CURRICULUM)}] REJECT {concept} - {last_error}")
            write_jsonl(REJECTS, {
                "category": category,
                "concept": concept,
                "reason": last_error,
            })
            continue

        accepted.append(record)
        write_jsonl(OUT, record)
        print(f"[{idx:02d}/{len(CURRICULUM)}] OK {concept} - {record['question'][:80]}")

    print("\n==============================")
    print("STATS TEACHER PILOT v0.1")
    print("==============================")
    print(f"Accepted: {len(accepted)} / {len(CURRICULUM)}")
    print(f"Approx. API calls: {calls}")
    print(f"Accepted file: {OUT}")
    print(f"Reject log:    {REJECTS}")

    print("\n--- FIRST 6 SAMPLES ---")
    for i, x in enumerate(accepted[:6], 1):
        print(f"\n[{i}] {x['category']} / {x['concept']}")
        print("Q:", x["question"])
        print("A:", x["answer"])
        print("Why:", x["explanation"])
        print("Common mistake:", x["common_mistake"])


if __name__ == "__main__":
    main()
