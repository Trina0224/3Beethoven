"""Unattended 3Beethoven statistics distillation run v0.1.

Purpose
-------
Generate a small response-distillation corpus from Llama 3.3 70B, QLoRA-train
Llama 3.2 3B Instruct, and evaluate the trained adapter on a frozen 16-question
held-out slice covering the two selected capability gaps:

- inference / hypothesis testing
- distributions / expectation

Designed for a Kaggle GPU batch run. No secrets are written to outputs.

Required Kaggle Secrets
-----------------------
HF_TOKEN
OPENROUTER_API_KEY

Important
---------
- Keep the frozen benchmark out of training data.
- Do not add Co-authored-by or AI/bot authorship metadata to commits.
- Do not include proprietary or NDA material.
"""

import gc
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

import requests
import torch
from datasets import Dataset
from kaggle_secrets import UserSecretsClient
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

TEACHER_MODEL = "meta-llama/llama-3.3-70b-instruct"
STUDENT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
RUN_DIR = Path("/kaggle/working/3beethoven_stats_flight_v0_1")
DATA_PATH = RUN_DIR / "teacher_train_v0_1.jsonl"
REJECT_PATH = RUN_DIR / "teacher_rejects_v0_1.jsonl"
ADAPTER_DIR = RUN_DIR / "adapter"
SUMMARY_PATH = RUN_DIR / "summary.json"
LOG_PATH = RUN_DIR / "trainer_log.json"

# 24 concepts x 5 independent scenarios = 120 teacher examples.
VARIANTS_PER_CONCEPT = 5
SEED = 224
random.seed(SEED)

CONCEPTS = [
    # inference / testing
    ("inference_testing", "p_value_interpretation"),
    ("inference_testing", "type_i_error"),
    ("inference_testing", "type_ii_error"),
    ("inference_testing", "statistical_power"),
    ("inference_testing", "alpha_tradeoff"),
    ("inference_testing", "confidence_interval_width"),
    ("inference_testing", "ci_and_two_sided_test"),
    ("inference_testing", "independent_t_test"),
    ("inference_testing", "paired_t_test"),
    ("inference_testing", "chi_square_independence"),
    ("inference_testing", "one_sample_mean_test"),
    ("inference_testing", "multiple_testing"),
    ("inference_testing", "practical_vs_statistical_significance"),
    ("inference_testing", "sample_size_power"),
    # distributions / expectation
    ("distributions_expectation", "bernoulli_mean_variance"),
    ("distributions_expectation", "binomial_mean_variance"),
    ("distributions_expectation", "poisson_mean_variance"),
    ("distributions_expectation", "linear_expectation"),
    ("distributions_expectation", "variance_scaling"),
    ("distributions_expectation", "variance_sum_independent"),
    ("distributions_expectation", "normal_68_95_997"),
    ("distributions_expectation", "uniform_expectation"),
    ("distributions_expectation", "exponential_memoryless"),
    ("distributions_expectation", "poisson_modeling"),
]

SYSTEM_TEACHER = """You are creating one high-quality statistics teaching example for
response distillation. Stay strictly within the requested concept. Accuracy and clean
statistical interpretation matter more than novelty.

Return exactly one valid JSON object with these string fields:
category, concept, question, answer, explanation, common_mistake.

Rules:
1. Use a fresh scenario and fresh numbers when numbers are useful.
2. The answer must be correct under standard introductory/intermediate statistics.
3. The explanation must state the decisive statistical principle.
4. The common_mistake must be a real misconception and must itself be accurate.
5. Do not use Bayes-theorem questions, regression/causality questions, or ML metrics.
6. Do not reproduce known benchmark wording. Create a materially different scenario.
7. Avoid unnecessary arithmetic complexity; the target is statistical reasoning.
8. Output JSON only, with no markdown fences.
"""

FROZEN_EVAL = [
    # distributions / expectation (original benchmark questions 9-16)
    ("distributions_expectation", "For a Bernoulli random variable X with parameter p, what is E[X]?", ["A. p", "B. p(1-p)", "C. 1-p", "D. p^2"], "A"),
    ("distributions_expectation", "For a Poisson random variable with parameter lambda, which statement is true?", ["A. Mean is lambda and variance is lambda^2", "B. Mean and variance are both lambda", "C. Mean is 1/lambda", "D. Variance is always 1"], "B"),
    ("distributions_expectation", "If X has mean 10 and Y=3X+2, what is E[Y]?", ["A. 12", "B. 20", "C. 32", "D. 36"], "C"),
    ("distributions_expectation", "Which distribution is commonly used to model the number of Bernoulli successes in n independent trials with fixed success probability p?", ["A. Normal", "B. Exponential", "C. Poisson", "D. Binomial"], "D"),
    ("distributions_expectation", "If X and Y are independent, Var(X)=4 and Var(Y)=9, what is Var(X+Y)?", ["A. 13", "B. 36", "C. 5", "D. 25"], "A"),
    ("distributions_expectation", "Which distribution has the memoryless property?", ["A. Normal", "B. Exponential", "C. Uniform", "D. Beta"], "B"),
    ("distributions_expectation", "For a standard normal distribution, approximately what percentage of observations lie within one standard deviation of the mean?", ["A. 50%", "B. 95%", "C. 68%", "D. 99.7%"], "C"),
    ("distributions_expectation", "If X is uniformly distributed on [0,10], what is E[X]?", ["A. 0", "B. 2.5", "C. 10", "D. 5"], "D"),
    # inference / testing (original benchmark questions 17-24)
    ("inference_testing", "A p-value of 0.03 means:", ["A. Assuming the null hypothesis is true, the observed result or something more extreme would occur with probability about 3%", "B. The null hypothesis has a 3% probability of being true", "C. The alternative hypothesis has a 97% probability of being true", "D. There is a 3% chance the study result is wrong"], "A"),
    ("inference_testing", "Which change generally makes a confidence interval narrower, all else equal?", ["A. Smaller sample size", "B. Larger sample size", "C. Higher confidence level", "D. Larger standard error"], "B"),
    ("inference_testing", "A Type I error occurs when:", ["A. A false null hypothesis is not rejected", "B. Both hypotheses are false", "C. A true null hypothesis is rejected", "D. The sample size is too small"], "C"),
    ("inference_testing", "A Type II error occurs when:", ["A. A true null is rejected", "B. A p-value is exactly 0.05", "C. A confidence interval is too wide", "D. A false null hypothesis is not rejected"], "D"),
    ("inference_testing", "If a 95% confidence interval for a mean difference excludes zero, what usually follows for a corresponding two-sided hypothesis test at alpha=0.05?", ["A. Reject the null hypothesis", "B. Increase alpha", "C. Accept the null with certainty", "D. No conclusion is possible"], "A"),
    ("inference_testing", "Statistical power is the probability of:", ["A. Rejecting a true null hypothesis", "B. Rejecting a false null hypothesis", "C. Failing to reject a true null", "D. Obtaining p=0"], "B"),
    ("inference_testing", "Which test is typically appropriate for comparing means of two independent groups when normality and equal-variance assumptions are reasonable?", ["A. Chi-square test", "B. Paired t-test", "C. Independent-samples t-test", "D. McNemar test"], "C"),
    ("inference_testing", "Increasing the confidence level from 95% to 99%, all else equal, generally makes the confidence interval:", ["A. Narrower", "B. Unchanged", "C. Centered at zero", "D. Wider"], "D"),
]


def jsonl_append(path: Path, obj):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def parse_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def teacher_call(api_key, category, concept, variant):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    user = f"""Category: {category}
Concept: {concept}
Scenario variant index: {variant}

Create one new teaching example for exactly this concept. Use a different context from common textbook examples when practical, but keep the statistical principle standard and unambiguous."""
    payload = {
        "model": TEACHER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_TEACHER},
            {"role": "user", "content": user},
        ],
        "temperature": 0.35,
        "max_tokens": 700,
    }
    r = requests.post(API_URL, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    return parse_json(r.json()["choices"][0]["message"]["content"])


def validate_teacher_record(obj, category, concept):
    required = ["category", "concept", "question", "answer", "explanation", "common_mistake"]
    if not isinstance(obj, dict):
        return False, "not an object"
    if any(not isinstance(obj.get(k), str) or not obj[k].strip() for k in required):
        return False, "missing required field"
    if obj["category"] != category or obj["concept"] != concept:
        return False, "wrong category/concept"
    if len(obj["question"]) < 30 or len(obj["explanation"]) < 80:
        return False, "too short"
    bad = ("bayes", "regression", "causality", "precision", "recall", "f1 score")
    blob = (obj["question"] + " " + obj["answer"] + " " + obj["explanation"]).lower()
    if any(x in blob for x in bad):
        return False, "off-scope concept"
    return True, ""


def build_corpus(api_key):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PATH.unlink(missing_ok=True)
    REJECT_PATH.unlink(missing_ok=True)
    accepted = []
    questions = set()
    api_calls = 0

    print(f"Generating {len(CONCEPTS) * VARIANTS_PER_CONCEPT} teacher examples...")
    for cidx, (category, concept) in enumerate(CONCEPTS, 1):
        for variant in range(1, VARIANTS_PER_CONCEPT + 1):
            record = None
            err = None
            for attempt in range(1, 3):
                try:
                    obj = teacher_call(api_key, category, concept, variant)
                    api_calls += 1
                    ok, reason = validate_teacher_record(obj, category, concept)
                    if not ok:
                        raise ValueError(reason)
                    norm_q = re.sub(r"\W+", " ", obj["question"].lower()).strip()
                    if norm_q in questions:
                        raise ValueError("exact duplicate")
                    obj["teacher_model"] = TEACHER_MODEL
                    obj["pipeline_version"] = "stats-flight-v0.1"
                    record = obj
                    break
                except Exception as e:
                    err = str(e)
                    time.sleep(1.0 * attempt)
            if record is None:
                jsonl_append(REJECT_PATH, {"category": category, "concept": concept, "variant": variant, "reason": err})
                print(f"[{cidx:02d}/{len(CONCEPTS)}] {concept} v{variant}: REJECT - {err}")
                continue
            questions.add(re.sub(r"\W+", " ", record["question"].lower()).strip())
            accepted.append(record)
            jsonl_append(DATA_PATH, record)
            print(f"[{cidx:02d}/{len(CONCEPTS)}] {concept} v{variant}: OK")

    print(f"Teacher corpus accepted: {len(accepted)} / {len(CONCEPTS) * VARIANTS_PER_CONCEPT}")
    print(f"Approx. teacher API calls: {api_calls}")
    return accepted, api_calls


def load_tokenizer(hf_token):
    tok = AutoTokenizer.from_pretrained(STUDENT_MODEL, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def make_training_dataset(records, tokenizer, max_length=768):
    rows = []
    for r in records:
        user = f"""Statistics question:\n{r['question']}\n\nAnswer the question and explain the decisive statistical principle. Also mention one common misconception to avoid."""
        assistant = f"""Answer: {r['answer']}\n\nExplanation: {r['explanation']}\n\nCommon misconception: {r['common_mistake']}"""
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
            tokenize=False,
            add_generation_prompt=False,
        )
        p = tokenizer(prompt_text, add_special_tokens=False, truncation=True, max_length=max_length)
        f = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)
        labels = [-100] * min(len(p["input_ids"]), len(f["input_ids"])) + f["input_ids"][len(p["input_ids"]):]
        labels = labels[: len(f["input_ids"])]
        if len(labels) < len(f["input_ids"]):
            labels += [-100] * (len(f["input_ids"]) - len(labels))
        rows.append({"input_ids": f["input_ids"], "attention_mask": f["attention_mask"], "labels": labels})
    return Dataset.from_list(rows)


class CausalCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)
        ids, masks, labels = [], [], []
        for x in features:
            pad = max_len - len(x["input_ids"])
            ids.append(x["input_ids"] + [self.tokenizer.pad_token_id] * pad)
            masks.append(x["attention_mask"] + [0] * pad)
            labels.append(x["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def train_adapter(records, hf_token):
    # Free any prior full-precision notebook model before loading QLoRA.
    if "model" in globals():
        try:
            del globals()["model"]
        except Exception:
            pass
    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = load_tokenizer(hf_token)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        token=hf_token,
        quantization_config=bnb,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dataset = make_training_dataset(records, tokenizer)
    split = dataset.train_test_split(test_size=max(8, int(len(dataset) * 0.08)), seed=SEED)

    args = TrainingArguments(
        output_dir=str(RUN_DIR / "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=True,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=CausalCollator(tokenizer),
    )
    result = trainer.train()
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    LOG_PATH.write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    return model, tokenizer, result, len(split["train"]), len(split["test"])


def evaluate_adapter(model, tokenizer):
    model.eval()
    rows = []
    for idx, (category, question, choices, expected) in enumerate(FROZEN_EVAL, 1):
        prompt = f"""Answer this statistics multiple-choice question.\n\nQuestion:\n{question}\n\n{chr(10).join(choices)}\n\nReply with ONLY the letter A, B, C, or D. Do not explain."""
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        m = re.search(r"\b([ABCD])\b", generated.upper())
        pred = m.group(1) if m else "INVALID"
        rows.append({"id": idx, "category": category, "expected": expected, "predicted": pred, "correct": pred == expected})
        print(f"EVAL {idx:02d}/16 {'OK' if pred == expected else 'MISS'} expected={expected} got={pred}")
    overall = sum(x["correct"] for x in rows) / len(rows)
    by_cat = {}
    for cat in sorted(set(x["category"] for x in rows)):
        vals = [x["correct"] for x in rows if x["category"] == cat]
        by_cat[cat] = sum(vals) / len(vals)
    return overall, by_cat, rows


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    secrets = UserSecretsClient()
    hf_token = secrets.get_secret("HF_TOKEN")
    openrouter_key = secrets.get_secret("OPENROUTER_API_KEY")

    start = time.time()
    records, api_calls = build_corpus(openrouter_key)
    if len(records) < 90:
        raise RuntimeError(f"Only {len(records)} teacher examples accepted; refusing to train below 90.")

    model, tokenizer, train_result, train_n, eval_n = train_adapter(records, hf_token)
    post_acc, post_by_cat, eval_rows = evaluate_adapter(model, tokenizer)

    summary = {
        "teacher_model": TEACHER_MODEL,
        "student_model": STUDENT_MODEL,
        "pipeline_version": "stats-flight-v0.1",
        "teacher_examples": len(records),
        "teacher_api_calls": api_calls,
        "train_examples": train_n,
        "validation_examples": eval_n,
        "train_loss": train_result.training_loss,
        "frozen_eval_questions": len(FROZEN_EVAL),
        "known_pretrain_targeted_baseline": 0.5625,
        "known_teacher_targeted_score": 1.0,
        "post_distillation_targeted_score": post_acc,
        "post_distillation_by_category": post_by_cat,
        "eval_rows": eval_rows,
        "elapsed_minutes": round((time.time() - start) / 60, 1),
        "adapter_dir": str(ADAPTER_DIR),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n========================================")
    print("3BEETHOVEN FLIGHT RUN COMPLETE")
    print("========================================")
    print(f"Teacher examples: {len(records)}")
    print(f"Training loss: {train_result.training_loss:.4f}")
    print("Targeted baseline before distillation: 56.25%")
    print(f"Targeted score after distillation:    {post_acc:.2%}")
    print(f"Teacher score on targeted benchmark:  100.00%")
    print(f"By category: {post_by_cat}")
    print(f"Elapsed: {summary['elapsed_minutes']} minutes")
    print(f"Outputs: {RUN_DIR}")


if __name__ == "__main__":
    main()
