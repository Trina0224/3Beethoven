"""Run one cell of the frozen two-start, two-seed comparison on Kaggle.

The script refuses draft protocols.  It validates online every configured number
of optimizer updates and stops at the first checkpoint passing every validation
gate.  The independent holdout is deliberately handled by a later frozen
evaluator, after all four selected checkpoints exist.
"""
import argparse
import gc
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

from flight_run_stats_v0_3 import STUDENT, read_json as read, save_json as save
from run_stats_v0_4 import BASE_REVISION, dataset
from run_stats_v0_17 import HASHES
from run_stats_v0_19 import evaluate
from stats_consolidation_pilot import build as build_candidate, digest
from stats_curriculum_v0_13 import KINDS

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/kaggle/working/3beethoven_stats_consolidation")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def category_rates(metrics):
    counts = metrics["category_counts"]
    return {key: value["correct"] / value["n"] for key, value in counts.items()}


def suite_metrics(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    return {
        "n": len(rows), "correct": sum(r["correct"] for r in rows),
        "pending": sum(r.get("review_required", False) for r in rows),
        "category_counts": {category: {"n": len(items), "correct": sum(r["correct"] for r in items)}
                            for category, items in sorted(grouped.items())},
    }


def gate_result(all_metrics, gate):
    checks = {}
    for key, minimum in gate["minimum_rate_by_suite_category"].items():
        suite, category = key.split(":", 1)
        item = all_metrics[suite]["category_counts"].get(category)
        checks[key] = bool(item and item["correct"] / item["n"] >= minimum)
    candidate = all_metrics["candidate"]
    checks["candidate:overall"] = candidate["correct"] / candidate["n"] >= gate["candidate_validation_overall_minimum_rate"]
    return {"passed": all(checks.values()), "checks": checks}


def validate_decision(config):
    if config.get("protocol_status") != "frozen_for_execution":
        raise RuntimeError("Protocol remains a draft; GPU training is blocked")
    training = config["training"]
    if training["starts"] != ["fresh_base_new_lora", "original_v15_continued_lora"]:
        raise RuntimeError("Unexpected comparison starts")
    if training["seeds"] != [2027, 31415]:
        raise RuntimeError("Unexpected seeds")
    if training["base_model"] != STUDENT or training["base_revision"] != BASE_REVISION:
        raise RuntimeError("Model identity drift")
    if training["v15_adapter_sha256"] != HASHES[15]:
        raise RuntimeError("Original v15 hash drift")
    if config["validation_gate"].get("status") != "frozen_for_execution":
        raise RuntimeError("Validation gate is not frozen")


def training_rows(verified, seed):
    replay = [r for r in read(REPO / "docs/STATS_V0_19_REPLAY_SOURCE.json") if r["category"] in KINDS]
    if len(replay) != 192:
        raise RuntimeError("Historical replay is not the frozen 192-row set")
    new = verified["train"]
    if not new or any(r.get("reference_conditioned") for r in new):
        raise RuntimeError("No verified independent teacher training rows")
    groups = defaultdict(list)
    for row in new:
        groups["new:" + row["story_id"]].append(row)
    for row in replay:
        groups["replay:" + row["source_id"]].append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    rows = []
    for key in keys:
        items = groups[key]
        rows.extend({"source_id": r["source_id"], "story_id": r.get("story_id", key),
                     "category": r["category"], "prompt": r["prompt"], "target": r["target"],
                     "source_group": key.split(":", 1)[0]} for r in items)
    return rows


def validation_suites():
    matrix = read(REPO / "docs/STATS_CONSOLIDATION_SELECTION_VALIDATION.json")
    if matrix.get("role") != "checkpoint_selection_only" or matrix.get("status") != "frozen_for_execution":
        raise RuntimeError("Selection-validation matrix is not frozen")
    return matrix["suites"]


def locate_v15(search_root, expected_hash):
    matches = [p.parent for p in Path(search_root).rglob("adapter_model.safetensors")
               if p.parent.parent.name == "3beethoven_stats_v0_15" and file_sha(p) == expected_hash]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one original v15 adapter, found {len(matches)}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", choices=("fresh_base_new_lora", "original_v15_continued_lora"), required=True)
    parser.add_argument("--seed", type=int, choices=(2027, 31415), required=True)
    parser.add_argument("--decision", type=Path, default=REPO / "docs/STATS_CONSOLIDATION_DECISION_DRAFT.json")
    parser.add_argument("--teacher-rows", type=Path, default=Path("/kaggle/working/3beethoven_stats_consolidation_teacher/verified_teacher_rows.json"))
    parser.add_argument("--v15-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    config = read(args.decision); validate_decision(config)
    verified = read(args.teacher_rows)
    if not verified:
        raise RuntimeError("Verified teacher rows are missing")
    train_rows = training_rows(verified, args.seed)
    suites = validation_suites()
    output = args.output_root / f"{args.start}_seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "start": args.start, "seed": args.seed, "decision_sha256": digest(config),
        "teacher_rows_sha256": file_sha(args.teacher_rows), "training_rows_sha256": digest(train_rows),
        "training_rows": len(train_rows), "new_teacher_rows": len(verified["train"]),
        "historical_replay_rows": 192, "base_revision": BASE_REVISION,
    }
    prior = read(output / "contract.json")
    if prior and prior != contract:
        raise RuntimeError("Existing output contract differs")
    save(output / "contract.json", contract)
    save(output / "training_order.json", train_rows)
    completed = read(output / "training_complete.json")
    if completed:
        print("RUN ALREADY COMPLETE", json.dumps(completed["selection"])); return
    existing_history = read(output / "validation_history.json", [])
    existing_pass = next((r for r in existing_history if r["gate"]["passed"]), None)
    if existing_pass:
        adapter = output / f"step_{existing_pass['step']}" / "adapter" / "adapter_model.safetensors"
        if not adapter.exists():
            raise RuntimeError("Passing validation record exists without its portable adapter")
        selection = {"selected_step": existing_pass["step"], "diagnostic_step": None,
                     "stop_reason": "recovered_first_full_validation_pass"}
        save(output / "selection.json", selection)
        save(output / "training_complete.json", {"global_steps": existing_pass["step"],
             "training_loss": None, "selection": selection,
             "note": "Recovered after validation pass; independent holdout and permanent MC remain deferred."})
        print("RECOVERED COMPLETE", json.dumps(selection)); return

    import torch
    from kaggle_secrets import UserSecretsClient
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              Trainer, TrainerCallback, TrainingArguments, set_seed)
    from flight_run_stats_v0_1 import CausalCollator
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU is required")
    training = config["training"]
    set_seed(args.seed)
    token = UserSecretsClient().get_secret("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT, revision=BASE_REVISION, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT, revision=BASE_REVISION, token=token, device_map={"": 0}, torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
    model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.start == "original_v15_continued_lora":
        v15 = locate_v15(args.v15_root, training["v15_adapter_sha256"])
        model = PeftModel.from_pretrained(model, v15, is_trainable=True)
        contract["parent_adapter_sha256"] = file_sha(v15 / "adapter_model.safetensors")
    else:
        model = get_peft_model(model, LoraConfig(
            r=training["lora_r"], lora_alpha=training["lora_alpha"], lora_dropout=training["lora_dropout"],
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        contract["parent_adapter_sha256"] = None
    save(output / "contract.json", contract)
    model.config.use_cache = False
    interval = training["validation_every_updates"]
    available_updates = math.ceil(len(train_rows) / training["gradient_accumulation_steps"])
    actual_max = min(training["max_updates"], available_updates)
    if actual_max < interval:
        raise RuntimeError("Accepted corpus is too small for one validation interval")

    class OnlineValidation(TrainerCallback):
        def __init__(self):
            self.evaluated = {r["step"] for r in read(output / "validation_history.json", [])}

        def run_validation(self, state, model):
            step = state.global_step
            if step in self.evaluated:
                return None
            self.evaluated.add(step)
            folder = output / f"step_{step}"
            folder.mkdir(exist_ok=True)
            result = {}
            for name, questions in suites.items():
                evaluate(model, tokenizer, questions, folder / f"{name}.json")
                result[name] = suite_metrics(read(folder / f"{name}.json"))
            gate = gate_result(result, config["validation_gate"])
            record = {"step": step, "metrics": result, "gate": gate,
                      "supervised_rows_seen_upper_bound": min(len(train_rows), step * training["gradient_accumulation_steps"])}
            history = [r for r in read(output / "validation_history.json", []) if r["step"] != step]
            history.append(record); history.sort(key=lambda r: r["step"])
            save(output / "validation_history.json", history)
            portable = folder / "adapter"
            model.save_pretrained(portable); tokenizer.save_pretrained(portable)
            save(folder / "adapter_hash.json", {"sha256": file_sha(portable / "adapter_model.safetensors")})
            print("ONLINE VALIDATION", json.dumps({"step": step, "gate": gate}), flush=True)
            return gate["passed"]

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if state.global_step % interval == 0 or state.global_step == actual_max:
                passed = self.run_validation(state, model)
                control.should_save = True
                if passed:
                    control.should_training_stop = True
            return control

    args_train = TrainingArguments(
        output_dir=str(output / "checkpoints"), max_steps=actual_max,
        per_device_train_batch_size=1, gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=training["learning_rate"], lr_scheduler_type=training["lr_scheduler_type"],
        warmup_steps=training["warmup_steps"], fp16=True, logging_steps=interval,
        save_strategy="no", report_to="none", remove_unused_columns=False,
        optim="paged_adamw_8bit", seed=args.seed, data_seed=args.seed, disable_tqdm=True)
    callback = OnlineValidation()
    trainer = Trainer(model=model, args=args_train, train_dataset=dataset(train_rows, tokenizer),
                      data_collator=CausalCollator(tokenizer), callbacks=[callback])
    checkpoints = sorted((output / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    result = trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)
    history = read(output / "validation_history.json", [])
    passed = next((r for r in history if r["gate"]["passed"]), None)
    selection = {"selected_step": passed["step"] if passed else None,
                 "diagnostic_step": None if passed else history[-1]["step"],
                 "stop_reason": "first_full_validation_pass" if passed else "update_cap_without_full_pass"}
    save(output / "selection.json", selection)
    save(output / "training_complete.json", {"global_steps": trainer.state.global_step,
         "actual_max_updates": actual_max, "available_one_pass_updates": available_updates,
         "training_loss": result.training_loss, "selection": selection,
         "note": "Independent holdout and permanent MC await the four-run frozen evaluator."})
    del trainer, model
    gc.collect(); torch.cuda.empty_cache()
    print("RUN COMPLETE", json.dumps(selection), flush=True)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
