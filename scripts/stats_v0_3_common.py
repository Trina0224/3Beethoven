"""Dependency-free, tested data and scoring helpers for the v0.3 rerun."""
import ast
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

SEED = 226
LETTERS = "ABCD"
CONCEPTS = ["poisson_mean_variance", "linear_expectation", "uniform_expectation",
            "type_i_error", "type_ii_error", "confidence_level_interval_width"]
CONTEXTS = ["a library", "a bakery", "a museum", "a train station", "an observatory",
            "a concert hall", "a garden", "a workshop", "a sports club", "a bookshop"]


def normalize_question(text):
    return re.sub(r"\W+", " ", text.casefold()).strip()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_frozen(filename):
    """Read the literal benchmark without importing GPU libraries or executing code."""
    tree = ast.parse(Path(filename).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FROZEN_EVAL" for t in node.targets):
            return [{"id": f"eval_{i:02d}", "category": cat, "question": q,
                     "choices": [re.sub(r"^[A-D]\.\s*", "", c) for c in cs],
                     "answer_letter": ans} for i, (cat, q, cs, ans) in enumerate(ast.literal_eval(node.value), 1)]
    raise ValueError("Frozen benchmark not found")


def make_curriculum():
    """60 fixed new questions. Code owns labels; the teacher supplies explanations."""
    records = []
    for ci, concept in enumerate(CONCEPTS):
        for v, context in enumerate(CONTEXTS):
            if concept == "poisson_mean_variance":
                lam = 13 + 2 * v
                q = f"At {context}, the count of arrivals in one hour is Poisson with mean {lam}. What is the variance of that hourly count?"
                correct, wrong = str(lam), [str(lam ** 2), f"sqrt({lam})", f"1/{lam}"]
                proof = f"For Poisson(lambda), both mean and variance equal lambda={lam}."
            elif concept == "linear_expectation":
                mean, a, b = 13 + v, 2 + v % 4, 7 + v
                q = f"A random daily quantity X at {context} has E[X]={mean}. A score is Y={a}X-{b}. What is E[Y]?"
                correct = str(a * mean - b)
                wrong = [str(a * mean + b), str(a * mean), str(mean - b)]
                proof = f"Linearity gives E[Y]={a}*{mean}-{b}={correct}; no distributional assumption is needed."
            elif concept == "uniform_expectation":
                low, high = 21 + 3 * v, 37 + 5 * v
                q = f"A waiting time X at {context} is continuously uniform between {low} and {high} minutes. What is E[X] in minutes?"
                correct, wrong = str((low + high) // 2), [str(low), str(high), str((high - low) // 2)]
                proof = f"The mean of Uniform(a,b) is (a+b)/2=({low}+{high})/2={correct}."
            elif concept in ("type_i_error", "type_ii_error"):
                value = 31 + v
                true_null = concept == "type_i_error"
                q = (f"An analyst at {context} tests H0: the population mean equals {value}. "
                     + ("In reality H0 is true, but the analyst rejects it. " if true_null else
                        "In reality H0 is false, but the analyst fails to reject it. ")
                     + "How should this decision be classified?")
                correct = "Type I error" if true_null else "Type II error"
                wrong = ["Type II error" if true_null else "Type I error", "Correct rejection", "Correct non-rejection"]
                proof = ("A Type I error rejects a true null hypothesis." if true_null else
                         "A Type II error fails to reject a false null hypothesis.")
            else:
                lo, hi = 80 + v, 95 + v % 4
                increasing = v % 2 == 0
                before, after = (lo, hi) if increasing else (hi, lo)
                q = (f"A report at {context} uses a two-sided normal-theory confidence interval for a mean. "
                     f"Keeping the data and standard error fixed, confidence changes from {before}% to {after}%. How does its width change?")
                correct = "It becomes wider" if increasing else "It becomes narrower"
                wrong = ["It becomes narrower" if increasing else "It becomes wider", "It stays the same", "It becomes zero"]
                proof = "Width is twice the critical value times standard error; higher confidence uses a larger critical value."
            position = (ci * 10 + v) % 4
            choices = list(wrong)
            choices.insert(position, correct)
            records.append({"id": f"{concept}_{v:02d}", "category": "distributions_expectation" if ci < 3 else "inference_testing",
                            "concept": concept, "question": q, "choices": choices,
                            "answer_letter": LETTERS[position], "reference_reason": proof})
    return records


def prompt_for(record, mode="letter"):
    choices = "\n".join(f"{letter}. {text}" for letter, text in zip(LETTERS, record["choices"]))
    if mode == "letter":
        return f"Answer this statistics multiple-choice question.\n\nQuestion:\n{record['question']}\n\n{choices}\n\nReply with ONLY the letter A, B, C, or D. Do not explain."
    if mode == "explain":
        return f"Statistics question:\n{record['question']}\n\n{choices}\n\nChoose A, B, C, or D and explain briefly. Start with 'Answer: <letter>'."
    raise ValueError(mode)


def parse_answer(text):
    """Only accept an explicit leading answer; never fish a letter from prose."""
    cleaned = text.strip().replace("**", "")
    match = re.match(r"^(?:Answer\s*:\s*)?([ABCD])(?:$|[\s.\):,])", cleaned, flags=re.I)
    return match.group(1).upper() if match else "INVALID"


def group_split(records):
    train, validation = [], []
    rng = random.Random(SEED)
    for concept in CONCEPTS:
        group = sorted((r for r in records if r["concept"] == concept), key=lambda r: r["id"])
        rng.shuffle(group)
        validation.extend(group[:2])
        train.extend(group[2:])
    return train, validation


def audit(records, benchmark):
    keys = [normalize_question(r["question"]) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate normalized training question")
    if set(keys) & {normalize_question(r["question"]) for r in benchmark}:
        raise ValueError("Exact train/evaluation overlap")
    for r in records:
        if len(r["choices"]) != 4 or len(set(r["choices"])) != 4 or r["answer_letter"] not in LETTERS:
            raise ValueError("Malformed choices or answer")
    counts = Counter(r["answer_letter"] for r in records)
    concepts = Counter(r["concept"] for r in records)
    if len(records) != 60 or any(counts[x] != 15 for x in LETTERS) or any(concepts[x] != 10 for x in CONCEPTS):
        raise ValueError("Corpus size/balance gate failed")
    return {"n": len(records), "positions": dict(counts), "concepts": dict(concepts),
            "exact_eval_overlap": 0, "corpus_sha256": digest(records)}


def score_rows(rows):
    return {"n": len(rows), "correct": sum(r["correct"] for r in rows),
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "invalid": sum(r["predicted"] == "INVALID" for r in rows),
            "predicted_positions": dict(Counter(r["predicted"] for r in rows))}
