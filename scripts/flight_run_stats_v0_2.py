"""3Beethoven statistics response-distillation flight run v0.2.

Targets the two observed v0.1 weaknesses: six persistent concept errors and
multiple-choice answer-position bias.  This module reuses the tested v0.1
training machinery, while generating a balanced multiple-choice curriculum and
evaluating on a fresh frozen set that is never sent to the teacher.
"""

import json
import random
import re
import time
from collections import Counter
from pathlib import Path

from datasets import Dataset
from kaggle_secrets import UserSecretsClient

import flight_run_stats_v0_1 as base


SEED = 225
random.seed(SEED)
RUN_DIR = Path("/kaggle/working/3beethoven_stats_flight_v0_2")
DATA_PATH = RUN_DIR / "teacher_train_v0_2.jsonl"
REJECT_PATH = RUN_DIR / "teacher_rejects_v0_2.jsonl"
SUMMARY_PATH = RUN_DIR / "summary.json"
TARGETS = [
    ("distributions_expectation", "poisson_mean_variance"),
    ("distributions_expectation", "linear_expectation"),
    ("distributions_expectation", "uniform_expectation"),
    ("inference_testing", "type_i_error"),
    ("inference_testing", "type_ii_error"),
    ("inference_testing", "confidence_level_interval_width"),
]
VARIANTS_PER_TARGET = 12
LETTERS = "ABCD"

SYSTEM_TEACHER = """Create one accurate multiple-choice statistics teaching example.
Return exactly one JSON object with fields: category, concept, question, choices,
answer_letter, explanation, common_mistake. choices must be an array of exactly
four non-empty strings in A/B/C/D order. Put the correct answer at the requested
letter. Make every distractor plausible but unambiguously wrong. Do not mention
the requested answer position. Avoid benchmark-like wording. JSON only."""

# Frozen before teacher generation. None of this wording is sent to the teacher.
FROZEN_EVAL = [
    ("distributions_expectation", "A Poisson variable has mean 7. What is its variance?", ["A. 49", "B. sqrt(7)", "C. 1/7", "D. 7"], "D"),
    ("distributions_expectation", "For X~Poisson(2.5), which pair gives (mean, variance)?", ["A. (2.5, 2.5)", "B. (2.5, 6.25)", "C. (1/2.5, 2.5)", "D. (2.5, sqrt(2.5))"], "A"),
    ("distributions_expectation", "If E[X]=4 and Y=5X-3, what is E[Y]?", ["A. 8", "B. 20", "C. 23", "D. 17"], "D"),
    ("distributions_expectation", "If E[Z]=-2, what is E[3Z+8]?", ["A. -14", "B. -6", "C. 2", "D. 6"], "C"),
    ("distributions_expectation", "X is uniform on [4,12]. What is E[X]?", ["A. 4", "B. 6", "C. 8", "D. 12"], "C"),
    ("distributions_expectation", "A continuous uniform variable spans [-3,5]. Its expectation is:", ["A. -3", "B. 0", "C. 2", "D. 1"], "D"),
    ("distributions_expectation", "Which statement about a Poisson(lambda) variable is correct?", ["A. Variance is always 1", "B. Mean is lambda squared", "C. Mean equals variance", "D. Mean is the reciprocal of variance"], "C"),
    ("distributions_expectation", "E[X]=6. Without assuming any distribution, E[2-4X] equals:", ["A. -22", "B. -16", "C. 22", "D. 26"], "A"),
    ("distributions_expectation", "For U uniform on [10,18], which value is E[U]?", ["A. 4", "B. 10", "C. 14", "D. 18"], "C"),
    ("distributions_expectation", "A Poisson count has variance 11. Its mean is:", ["A. sqrt(11)", "B. 121", "C. Cannot be known", "D. 11"], "D"),
    ("distributions_expectation", "If E[A]=1.5, E[10A+1] is:", ["A. 11.5", "B. 15", "C. 16", "D. 25"], "C"),
    ("distributions_expectation", "X is uniform on [a,b]. Its expectation is:", ["A. b-a", "B. (a+b)/2", "C. ab/2", "D. 1/(b-a)"], "B"),
    ("inference_testing", "Rejecting a null hypothesis that is actually true is:", ["A. Type I error", "B. Type II error", "C. high power", "D. correct rejection"], "A"),
    ("inference_testing", "Failing to reject a false null hypothesis is:", ["A. Type I error", "B. Type II error", "C. correct acceptance with certainty", "D. significance"], "B"),
    ("inference_testing", "Holding data fixed, changing a 95% CI to 99% generally makes it:", ["A. narrower", "B. wider", "C. unchanged", "D. centered at zero"], "B"),
    ("inference_testing", "Which action generally lowers Type I error probability?", ["A. Raise alpha", "B. Lower alpha", "C. Lower sample size", "D. Increase variance"], "B"),
    ("inference_testing", "A Type II error concerns which state of the null?", ["A. It is true and rejected", "B. It is false and not rejected", "C. It is true and not rejected", "D. It is false and rejected"], "B"),
    ("inference_testing", "All else equal, a 90% interval compared with a 95% interval is usually:", ["A. wider", "B. identical", "C. undefined", "D. narrower"], "D"),
    ("inference_testing", "The conventional symbol alpha denotes the probability of:", ["A. Type I error", "B. Type II error", "C. a false null", "D. statistical power"], "A"),
    ("inference_testing", "Not rejecting H0 even though H0 is false is best described as:", ["A. correct decision", "B. Type I error", "C. Type II error", "D. p-hacking"], "C"),
    ("inference_testing", "Why does higher confidence typically widen an interval?", ["A. It uses a larger critical value", "B. It reduces the sample mean", "C. It removes variability", "D. It sets standard error to zero"], "A"),
    ("inference_testing", "If alpha decreases from .05 to .01, Type I error risk:", ["A. increases", "B. decreases", "C. becomes Type II error", "D. must equal zero"], "B"),
    ("inference_testing", "Which is a correct detection when H0 is false?", ["A. Reject H0", "B. Fail to reject H0", "C. Commit Type I error", "D. Widen every CI"], "A"),
    ("inference_testing", "With the same sample and method, which interval is widest?", ["A. 80%", "B. 90%", "C. 95%", "D. 99%"], "D"),
]


def append_jsonl(path, obj):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def teacher_call(api_key, category, concept, variant, answer_letter, attempt=1):
    scenario_seed = (SEED * 1009 + variant * 97 + attempt * 31) % 10000
    user = (
        f"Category: {category}\nConcept: {concept}\nVariant: {variant}\n"
        f"Required correct-answer position: {answer_letter}\n"
        f"Scenario seed: {scenario_seed}\n"
        "Make this question materially different from standard textbook one-liners. "
        "Use the scenario seed to choose fresh names, context, and numerical values; "
        "the seed itself must not appear in the question."
    )
    payload = {
        "model": base.TEACHER_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_TEACHER}, {"role": "user", "content": user}],
        "temperature": 0.35,
        "max_tokens": 700,
    }
    response = base.requests.post(
        base.API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=90,
    )
    response.raise_for_status()
    return base.parse_json(response.json()["choices"][0]["message"]["content"])


def validate(record, category, concept, answer_letter):
    required = ("category", "concept", "question", "choices", "answer_letter", "explanation", "common_mistake")
    if not isinstance(record, dict) or any(key not in record for key in required):
        return False, "missing field"
    if record["category"] != category or record["concept"] != concept:
        return False, "wrong category/concept"
    if record["answer_letter"] != answer_letter:
        return False, "wrong answer position"
    if not isinstance(record["choices"], list) or len(record["choices"]) != 4:
        return False, "choices must contain four items"
    if any(not isinstance(x, str) or not x.strip() for x in record["choices"]):
        return False, "empty choice"
    if len(record["question"]) < 20 or len(record["explanation"]) < 60:
        return False, "content too short"
    return True, ""


def build_corpus(api_key):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    accepted = []
    if DATA_PATH.exists():
        with DATA_PATH.open("r", encoding="utf-8") as handle:
            accepted = [json.loads(line) for line in handle if line.strip()]
        print(f"Resuming with {len(accepted)} accepted examples")

    questions = {
        re.sub(r"\\W+", " ", item.get("question", "").lower()).strip()
        for item in accepted
    }
    position_counts = Counter(item["answer_letter"] for item in accepted)
    concept_counts = Counter(item["concept"] for item in accepted)
    target_per_concept = 10
    target_per_position = 15
    max_paid_calls = 120
    paid_calls = 0
    sequence = 1000

    while any(concept_counts[concept] < target_per_concept for _, concept in TARGETS):
        progress = False
        for category, concept in TARGETS:
            if concept_counts[concept] >= target_per_concept:
                continue
            eligible = [letter for letter in LETTERS if position_counts[letter] < target_per_position]
            letter = min(eligible or list(LETTERS), key=lambda x: (position_counts[x], x))
            sequence += 1
            record, reason = None, "unknown"
            for attempt in range(1, 4):
                if paid_calls >= max_paid_calls:
                    raise RuntimeError(
                        f"Supplemental call cap reached; preserved n={len(accepted)} "
                        f"positions={dict(position_counts)} concepts={dict(concept_counts)}"
                    )
                paid_calls += 1
                try:
                    candidate = teacher_call(api_key, category, concept, sequence, letter, attempt)
                    ok, reason = validate(candidate, category, concept, letter)
                    normalized = re.sub(r"\\W+", " ", candidate.get("question", "").lower()).strip()
                    if not ok:
                        raise ValueError(reason)
                    if normalized in questions:
                        raise ValueError("duplicate question")
                    candidate.update(
                        teacher_model=base.TEACHER_MODEL,
                        pipeline_version="stats-flight-v0.2",
                    )
                    record = candidate
                    questions.add(normalized)
                    break
                except Exception as exc:
                    reason = str(exc)
                    time.sleep(attempt)
            if record is None:
                append_jsonl(
                    REJECT_PATH,
                    {"category": category, "concept": concept, "variant": sequence, "reason": reason},
                )
                print(f"[supplement {paid_calls:03d}] REJECT {concept} {letter}: {reason}")
            else:
                accepted.append(record)
                append_jsonl(DATA_PATH, record)
                position_counts[letter] += 1
                concept_counts[concept] += 1
                progress = True
                print(
                    f"[supplement {paid_calls:03d}] OK {concept} answer={letter} "
                    f"total={len(accepted)}/60"
                )
        if not progress and paid_calls >= max_paid_calls:
            break

    print("Accepted:", len(accepted), "answer positions:", dict(position_counts))
    print("Concept counts:", dict(concept_counts))
    if (
        len(accepted) < 60
        or min(concept_counts.values(), default=0) < target_per_concept
        or max(position_counts.values(), default=0) - min(position_counts.values(), default=0) > 2
    ):
        raise RuntimeError(
            f"Corpus failed size/balance gate: n={len(accepted)}, "
            f"positions={dict(position_counts)}, concepts={dict(concept_counts)}"
        )
    return accepted, position_counts


def make_training_dataset(records, tokenizer, max_length=768):
    rows = []
    for record in records:
        choices = "\n".join(f"{letter}. {text}" for letter, text in zip(LETTERS, record["choices"]))
        user = f"Statistics question:\n{record['question']}\n\n{choices}\n\nChoose A, B, C, or D and explain briefly."
        assistant = f"Answer: {record['answer_letter']}\n\nExplanation: {record['explanation']}\n\nCommon misconception: {record['common_mistake']}"
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}], tokenize=False, add_generation_prompt=False)
        p = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_length)
        f = tokenizer(full, add_special_tokens=False, truncation=True, max_length=max_length)
        labels = ([-100] * min(len(p["input_ids"]), len(f["input_ids"])) + f["input_ids"][len(p["input_ids"]):])[:len(f["input_ids"])]
        rows.append({"input_ids": f["input_ids"], "attention_mask": f["attention_mask"], "labels": labels})
    return Dataset.from_list(rows)


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    base.RUN_DIR = RUN_DIR
    base.ADAPTER_DIR = RUN_DIR / "adapter"
    base.LOG_PATH = RUN_DIR / "trainer_log.json"
    base.SUMMARY_PATH = SUMMARY_PATH
    base.FROZEN_EVAL = FROZEN_EVAL
    base.make_training_dataset = make_training_dataset
    secrets = UserSecretsClient()
    records, positions = build_corpus(secrets.get_secret("OPENROUTER_API_KEY"))
    model, tokenizer, train_result, train_n, validation_n = base.train_adapter(records, secrets.get_secret("HF_TOKEN"))
    score, by_category, rows = base.evaluate_adapter(model, tokenizer)
    summary = {
        "pipeline_version": "stats-flight-v0.2",
        "teacher_model": base.TEACHER_MODEL,
        "student_model": base.STUDENT_MODEL,
        "teacher_examples": len(records),
        "answer_position_counts": dict(positions),
        "train_examples": train_n,
        "validation_examples": validation_n,
        "training_loss": train_result.training_loss,
        "fresh_frozen_eval_questions": len(FROZEN_EVAL),
        "post_distillation_score": score,
        "post_distillation_by_category": by_category,
        "eval_rows": rows,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("3BEETHOVEN V0.2 COMPLETE", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
