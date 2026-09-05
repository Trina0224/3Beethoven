"""Bounded statistics repair experiment: fixed labels, same-exam controls, durable ledger.

Run with CUDA_VISIBLE_DEVICES=0. No secrets are saved. --stage generate is CPU-safe.
v0.2's exposed 24 questions are a development benchmark, not a new blind test.
"""
import argparse
import importlib.metadata
import json
import os
import shutil
import zipfile
from pathlib import Path

from stats_v0_3_common import (SEED, audit, digest, group_split, make_curriculum,
                             parse_answer, parse_teacher, prompt_for, read_frozen, score_rows)

TEACHER = "meta-llama/llama-3.3-70b-instruct"
STUDENT = "meta-llama/Llama-3.2-3B-Instruct"
MODES = {"legacy4": ("letter", 4), "letter16": ("letter", 16), "explain64": ("explain", 64)}
TRAINING = {"epochs": 3, "learning_rate": 5e-5, "lora_r": 16, "lora_alpha": 32,
            "batch_size": 1, "gradient_accumulation": 8, "max_length": 768,
            "seed": SEED, "primary_metric": "letter16"}


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def append(path, obj):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TeacherClient:
    def __init__(self, root, key, max_calls):
        self.root, self.key, self.max_calls = root, key, max_calls
        self.ledger = root / "api_ledger.jsonl"
        (root / "api_cache").mkdir(exist_ok=True)

    def stats(self):
        rows = read_jsonl(self.ledger)
        starts = [r for r in rows if r["event"] == "started"]
        completed = [r for r in rows if r["event"] == "response"]
        return {"attempted_calls": len(starts), "responses": len(completed),
                "reported_cost_usd": sum((r.get("usage", {}).get("cost") or 0) for r in completed),
                "responses_without_cost": sum(r.get("usage", {}).get("cost") is None for r in completed)}

    def call(self, tag, messages, max_tokens=400, json_mode=False):
        payload = {"model": TEACHER, "messages": messages, "temperature": 0,
                   "max_tokens": max_tokens,
                   "provider": {"max_price": {"prompt": 1, "completion": 2, "request": 0},
                                "enforce_distillable_text": True}}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        signature = digest(payload)
        cache = self.root / "api_cache" / (tag + ".json")
        prior = read_json(cache)
        if prior:
            if prior["request_sha256"] != signature:
                raise RuntimeError("Cached request differs; use a new run directory")
            return prior["text"]
        # Count attempts BEFORE the request. Never reset on restart.
        history = read_jsonl(self.ledger)
        if any(r["event"] == "started" and r["tag"] == tag for r in history):
            raise RuntimeError(f"Unresolved earlier request {tag}; refusing automatic duplicate billing")
        if self.stats()["attempted_calls"] >= self.max_calls:
            raise RuntimeError("Persistent API call cap reached; existing data preserved")
        # Conservative byte bound, plus limited output and provider price ceiling.
        if sum(len(m["content"].encode()) for m in messages) > 4000 or max_tokens > 400:
            raise RuntimeError("Request exceeds bounded token/byte policy")
        import requests
        append(self.ledger, {"event": "started", "tag": tag, "request_sha256": signature})
        try:
            response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                     headers={"Authorization": "Bearer " + self.key},
                                     json=payload, timeout=90)
        except requests.RequestException as exc:
            append(self.ledger, {"event": "network_error", "tag": tag, "type": type(exc).__name__})
            raise RuntimeError("Teacher network failure; stopped without automatic retry") from None
        if response.status_code != 200:
            append(self.ledger, {"event": "http_error", "tag": tag, "status": response.status_code})
            raise RuntimeError(f"Teacher HTTP {response.status_code}; stopped; secrets and response body omitted")
        result = response.json()
        if "choices" not in result:
            raise RuntimeError("Teacher returned no choices; stopped")
        text = result["choices"][0]["message"]["content"]
        if not isinstance(text, str):
            raise RuntimeError("Teacher returned non-text content")
        save_json(cache, {"request_sha256": signature, "request": payload, "text": text,
                          "usage": result.get("usage", {}), "model": result.get("model"),
                          "provider": result.get("provider"), "id": result.get("id")})
        append(self.ledger, {"event": "response", "tag": tag, "usage": result.get("usage", {})})
        return text


def prepare_data(root, client, plan, benchmark):
    teacher_file = root / "teacher_eval.json"
    teacher_rows = read_json(teacher_file, [])
    done = {r["id"] for r in teacher_rows}
    for r in benchmark:
        if r["id"] in done:
            continue
        raw = client.call("eval_" + r["id"], [{"role": "user", "content": prompt_for(r)}], max_tokens=16)
        pred = parse_answer(raw)
        teacher_rows.append({"id": r["id"], "category": r["category"], "expected": r["answer_letter"],
                             "predicted": pred, "correct": pred == r["answer_letter"], "raw": raw})
        save_json(teacher_file, teacher_rows)
        print(f"TEACHER EVAL {len(teacher_rows)}/24 {pred}", flush=True)
    # This evaluation cache is never read by the training-data builder.
    if score_rows(teacher_rows)["accuracy"] < 0.8:
        raise RuntimeError("Teacher development-benchmark score below 80%; generation stopped")
    data_path = root / "teacher_train.jsonl"
    records = read_jsonl(data_path)
    done = {r["id"] for r in records}
    for item in plan:
        if item["id"] in done:
            continue
        accepted = None
        for attempt in range(2):
            instructions = ("Solve the supplied statistics question independently. Return JSON with string fields "
                            "answer_letter, explanation, common_mistake. Explain the decisive principle in "
                            "two concise sentences. The misconception must be explicitly described as incorrect.")
            tag = f"train_{item['id']}_{attempt}"
            cached = read_json(root / "api_cache" / (tag + ".json"))
            # Preserve earlier paid responses even when repairing prompt formatting.
            if cached:
                raw = cached["text"]
            else:
                question = prompt_for(item, "explain").split("\n\nChoose A, B, C, or D")[0]
                raw = client.call(tag, [{"role": "system", "content": instructions},
                                       {"role": "user", "content": question + "\n\nReturn only the requested JSON object."}], json_mode=True)
            try:
                obj = parse_teacher(raw)
                if obj.get("answer_letter") != item["answer_letter"]:
                    raise ValueError("Teacher answer disagrees with deterministic reference")
                for field, length in (("explanation", 60), ("common_mistake", 15)):
                    if not isinstance(obj.get(field), str) or len(obj[field]) < length:
                        raise ValueError("Invalid explanation schema")
                accepted = {**item, "explanation": obj["explanation"], "common_mistake": obj["common_mistake"],
                            "teacher_model": TEACHER, "pipeline_version": "stats-flight-v0.3"}
                break
            except (ValueError, TypeError) as exc:
                append(root / "teacher_rejects.jsonl", {"id": item["id"], "attempt": attempt, "reason": str(exc)})
        if accepted is None:
            raise RuntimeError(f"Teacher failed validation twice for {item['id']}; partial corpus preserved")
        append(data_path, accepted)
        records.append(accepted)
        print(f"CORPUS {len(records)}/60 {item['id']} {client.stats()}", flush=True)
    report = audit(records, benchmark)
    save_json(root / "data_audit.json", report)
    return records


def evaluate(model, tokenizer, benchmark, path):
    import torch
    model.eval()
    model.config.use_cache = True
    rows = read_json(path, [])
    done = {(r["id"], r["mode"]) for r in rows}
    for name, (mode, max_tokens) in MODES.items():
        for r in benchmark:
            if (r["id"], name) in done:
                continue
            chat = tokenizer.apply_chat_template([{"role": "user", "content": prompt_for(r, mode)}],
                                                 tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(chat, add_special_tokens=False, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                result = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                        pad_token_id=tokenizer.eos_token_id)
            new_tokens = result[0][inputs["input_ids"].shape[-1]:]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            predicted = parse_answer(raw)
            rows.append({"id": r["id"], "category": r["category"], "mode": name,
                         "expected": r["answer_letter"], "predicted": predicted,
                         "correct": predicted == r["answer_letter"], "raw": raw,
                         "generated_tokens": len(new_tokens), "hit_token_limit": len(new_tokens) == max_tokens})
            save_json(path, rows)
        print(path.stem, name, score_rows([r for r in rows if r["mode"] == name]), flush=True)
    return rows


def training_dataset(records, tokenizer):
    from datasets import Dataset
    rows = []
    for r in records:
        for mode in ("letter", "explain"):
            assistant = (r["answer_letter"] if mode == "letter" else
                         f"Answer: {r['answer_letter']}\n\nExplanation: {r['explanation']}\n\nCommon misconception: {r['common_mistake']}")
            messages = [{"role": "user", "content": prompt_for(r, mode)}]
            prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": assistant}],
                                                  tokenize=True, add_generation_prompt=False)
            if full[:len(prefix)] != prefix:
                raise RuntimeError("Chat-template boundary mismatch")
            if len(full) > TRAINING["max_length"] or len(full) <= len(prefix):
                raise RuntimeError("Training example would be truncated or have no supervised tokens")
            rows.append({"input_ids": full, "attention_mask": [1] * len(full),
                         "labels": [-100] * len(prefix) + full[len(prefix):]})
    return Dataset.from_list(rows)


def train_and_compare(root, records, benchmark, hf_token):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    from flight_run_stats_v0_1 import CausalCollator
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Run in a fresh process with CUDA_VISIBLE_DEVICES=0; exactly one visible GPU required")
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(STUDENT, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT, token=hf_token, device_map={"": 0}, torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
    save_json(root / "model_environment.json", {"student_revision": getattr(model.config, "_commit_hash", None),
               "gpu": torch.cuda.get_device_name(0), "gpu_count": 1,
               "packages": {p: importlib.metadata.version(p) for p in
                            ("torch", "transformers", "peft", "bitsandbytes", "datasets", "accelerate")}})
    model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
    baseline = evaluate(model, tokenizer, benchmark, root / "baseline_eval.json")
    train, val = group_split(records)
    save_json(root / "split.json", {"train_ids": [r["id"] for r in train], "validation_ids": [r["id"] for r in val]})
    adapter_path = root / "adapter"
    complete = read_json(root / "training_complete.json")
    if complete:
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                              task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        model.config.use_cache = False
        args = TrainingArguments(output_dir=str(root / "checkpoints"), num_train_epochs=TRAINING["epochs"],
                                 per_device_train_batch_size=1, per_device_eval_batch_size=1,
                                 gradient_accumulation_steps=8, learning_rate=TRAINING["learning_rate"],
                                 warmup_steps=2, lr_scheduler_type="cosine", fp16=True, logging_steps=5,
                                 eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
                                 load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
                                 report_to="none", remove_unused_columns=False, optim="paged_adamw_8bit", seed=SEED)
        trainer = Trainer(model=model, args=args, train_dataset=training_dataset(train, tokenizer),
                          eval_dataset=training_dataset(val, tokenizer), data_collator=CausalCollator(tokenizer))
        checkpoints = sorted((root / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        result = trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        complete = {"training_loss": result.training_loss, "train_questions": len(train), "validation_questions": len(val),
                    "train_sequences": len(train) * 2, "best_validation_loss": trainer.state.best_metric,
                    "best_checkpoint": trainer.state.best_model_checkpoint}
        save_json(root / "trainer_log.json", trainer.state.log_history)
        save_json(root / "training_complete.json", complete)
    post = evaluate(model, tokenizer, benchmark, root / "post_eval.json")
    summary = {"pipeline_version": "stats-flight-v0.3", "teacher_model": TEACHER, "student_model": STUDENT,
               "training": complete, "config": TRAINING, "benchmark_sha256": digest(benchmark),
               "benchmark_status": "Previously exposed v0.2 development set; not a new blind test",
               "teacher": score_rows(read_json(root / "teacher_eval.json")), "comparisons": {}}
    for mode in MODES:
        before = [r for r in baseline if r["mode"] == mode]
        after = [r for r in post if r["mode"] == mode]
        by_id = {r["id"]: r for r in before}
        summary["comparisons"][mode] = {"baseline": score_rows(before), "distilled": score_rows(after),
            "wrong_to_right": sum(r["correct"] and not by_id[r["id"]]["correct"] for r in after),
            "right_to_wrong": sum(not r["correct"] and by_id[r["id"]]["correct"] for r in after)}
    return summary


def package(root):
    files = [p for p in root.rglob("*") if p.is_file() and "checkpoints" not in p.parts and p.name not in ("manifest.json",) and not p.name.endswith(".tmp")]
    import hashlib
    manifest = []
    for p in sorted(files):
        with p.open("rb") as handle:
            sha = hashlib.file_digest(handle, "sha256").hexdigest()
        manifest.append({"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": sha})
    save_json(root / "manifest.json", manifest)
    archive = root.parent / (root.name + ".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files + [root / "manifest.json"]:
            z.write(p, str(p.relative_to(root)))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise RuntimeError("Archive integrity check failed")
    print("ARCHIVE VERIFIED", archive, archive.stat().st_size, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("/kaggle/working/3beethoven_stats_flight_v0_3"))
    parser.add_argument("--max-calls", type=int, default=120)
    parser.add_argument("--stage", choices=["generate", "train", "all", "package"], default="all")
    args = parser.parse_args()
    if not 1 <= args.max_calls <= 120:
        raise ValueError("User-approved limit is at most 120 calls")
    root = args.run_dir
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == "package":
        package(root)
        return
    plan = make_curriculum()
    benchmark = read_frozen(Path(__file__).with_name("flight_run_stats_v0_2.py"))
    audit(plan, benchmark)
    protocol = {"curriculum_sha256": digest(plan), "benchmark_sha256": digest(benchmark), "training": TRAINING,
                "max_calls": args.max_calls, "teacher": TEACHER, "student": STUDENT}
    previous = read_json(root / "protocol.json")
    if previous and previous != protocol:
        raise RuntimeError("Protocol changed; use a new output directory")
    save_json(root / "protocol.json", protocol)
    save_json(root / "curriculum.json", plan)
    save_json(root / "development_benchmark.json", benchmark)
    (root / "source").mkdir(exist_ok=True)
    for name in ("flight_run_stats_v0_3.py", "stats_v0_3_common.py", "flight_run_stats_v0_1.py", "flight_run_stats_v0_2.py"):
        shutil.copy2(Path(__file__).with_name(name), root / "source" / name)
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    client = TeacherClient(root, secrets.get_secret("OPENROUTER_API_KEY"), args.max_calls)
    if args.stage in ("all", "generate"):
        records = prepare_data(root, client, plan, benchmark)
        package(root)
    else:
        records = read_jsonl(root / "teacher_train.jsonl")
        audit(records, benchmark)
    if args.stage in ("all", "train"):
        existing = read_json(root / "summary.json")
        summary = existing or train_and_compare(root, records, benchmark, secrets.get_secret("HF_TOKEN"))
        summary["api_usage"] = client.stats()
        save_json(root / "summary.json", summary)
        package(root)
        print("3BEETHOVEN V0.3 COMPLETE", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
