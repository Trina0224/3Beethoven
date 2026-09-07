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

from flight_run_stats_v0_3 import STUDENT, package, read_json as read, save_json as save
from run_stats_v0_4 import BASE_REVISION, dataset
from run_stats_v0_17 import HASHES
from stats_consolidation_eval import evaluate
from stats_consolidation_pilot import build as build_candidate, digest, assert_execution_released
from stats_consolidation_grader import score, grader_fingerprint
from stats_curriculum_v0_13 import KINDS

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/kaggle/working/3beethoven_stats_consolidation")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_immutable_contract(path, contract):
    prior = read(path)
    if prior and prior != contract:
        raise RuntimeError("Existing output contract differs")
    save(path, contract)


def category_rates(metrics):
    counts = metrics["category_counts"]
    return {key: value["correct"] / value["n"] for key, value in counts.items()}


def suite_metrics(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    return {
        "n": len(rows), "correct": sum(r["primary_correct"] for r in rows),
        "pending": sum(r.get("review_required", False) for r in rows),
        "category_counts": {category: {"n": len(items), "correct": sum(r["primary_correct"] for r in items),
                                        "depth3_n": sum(r.get("depth") == 3 for r in items),
                                        "depth3_correct": sum(r["primary_correct"] for r in items if r.get("depth") == 3)}
                            for category, items in sorted(grouped.items())},
    }


def gate_result(all_metrics, baseline, gate):
    checks = {"no_pending": sum(m["pending"] for m in all_metrics.values()) == 0}
    for suite in ("old", "chain"):
        rule = gate["suites"][suite]
        checks[suite + ":total"] = all_metrics[suite]["correct"] >= max(0, baseline[suite]["correct"] - rule["allowed_total_loss"])
        for category, base in baseline[suite]["category_counts"].items():
            current = all_metrics[suite]["category_counts"][category]
            checks[f"{suite}:{category}"] = current["correct"] >= max(0, base["correct"] - rule["allowed_per_category_loss"])
            if suite == "chain":
                checks[f"{suite}:{category}:depth3"] = current["depth3_correct"] >= base["depth3_correct"]
    event = all_metrics["event"]
    erule = gate["suites"]["event"]
    checks["event:total"] = event["correct"] >= erule["minimum_correct_total"]
    for category, item in event["category_counts"].items():
        checks["event:" + category] = item["correct"] >= erule["minimum_correct_per_category"]
    return {"passed": all(checks.values()), "checks": checks}


def validate_decision(config):
    assert_execution_released(config)
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


def validate_verified_rows(verified, candidate_stories):
    expected = {q["id"]: (story, q) for story in candidate_stories for q in story["questions"]}
    seen = set()
    for split in ("train", "validation"):
        for row in verified.get(split, []):
            if row["source_id"] in seen or row["source_id"] not in expected:
                raise RuntimeError("Teacher row is duplicate or unknown")
            seen.add(row["source_id"])
            story, q = expected[row["source_id"]]
            if story["split"] != split or row["question_sha256"] != digest(q):
                raise RuntimeError("Teacher row split or question hash drift")
            if row.get("reference_conditioned") or not row.get("teacher_raw") or not row.get("teacher_raw_sha256"):
                raise RuntimeError("Teacher row provenance is incomplete")
            if digest(row["teacher_raw"]) != row["teacher_raw_sha256"]:
                raise RuntimeError("Teacher raw hash drift")
            current = score("Expression: " + row["teacher_raw"], q)
            if not current["primary_correct"] or row["target"] != "Expression: " + current["normalized_expression"]:
                raise RuntimeError("Teacher target disagrees with freshly validated raw answer")
            judged = row.get("validation", {})
            if not (judged.get("math_correct") is True and judged.get("executable") is True):
                raise RuntimeError("Unverified teacher row")
    if len(verified.get("train", [])) > 198 or len(verified.get("validation", [])) > 70:
        raise RuntimeError("Accepted teacher corpus exceeds the frozen split limits")


def token_accounting(rows, tokenizer):
    prepared = dataset(rows, tokenizer)
    totals = {"rows": len(prepared), "input_tokens": 0, "supervised_tokens": 0}
    for row in prepared:
        totals["input_tokens"] += len(row["input_ids"])
        totals["supervised_tokens"] += sum(value != -100 for value in row["labels"])
    return totals


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
    parser.add_argument("--teacher-gate", type=Path, default=Path("/kaggle/working/3beethoven_stats_consolidation_teacher/full_gate.json"))
    parser.add_argument("--v15-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--pilot-student", action="store_true", help="Explicit separately preregistered filtered-pilot diagnostic only")
    args = parser.parse_args()
    config = read(args.decision); validate_decision(config)
    verified = read(args.teacher_rows)
    if not verified:
        raise RuntimeError("Verified teacher rows are missing")
    teacher_gate = read(args.teacher_gate, {})
    if not teacher_gate.get("passed"):
        raise RuntimeError("A persisted passing full teacher gate is required")
    required_scope = "filtered_pilot_student" if args.pilot_student else "full"
    if args.pilot_student:
        if config.get('first_batch',{}).get('mode') != 'filtered_pilot_student_no_promotion':
            raise RuntimeError('Filtered-pilot training is not authorized by this decision')
        if len(verified.get('train',[])) < 60 or teacher_gate.get('full_teacher_gate_passed') is not False:
            raise RuntimeError('Filtered-pilot quality/scope contract is missing')
        planned_archetypes={s['archetype'] for s in build_candidate()[0]}
        trained_ids={r['source_id'] for r in verified['train']}
        represented={s['archetype'] for s in build_candidate()[0] if any(q['id'] in trained_ids for q in s['questions'])}
        if represented != planned_archetypes:
            raise RuntimeError('Filtered pilot lacks a training archetype')
    if (teacher_gate.get("grader_fingerprint") != grader_fingerprint()
            or teacher_gate.get("verified_rows_sha256") != digest(verified)
            or teacher_gate.get("decision_sha256") != digest(config)
            or teacher_gate.get("request_sha256") != digest(build_candidate()[1])
            or teacher_gate.get("scope") != required_scope):
        raise RuntimeError("Teacher gate belongs to a different data/grader contract")
    candidate_stories = build_candidate()[0]
    validate_verified_rows(verified, candidate_stories)
    train_rows = training_rows(verified, args.seed)
    suites = validation_suites()
    v15 = locate_v15(args.v15_root, config["training"]["v15_adapter_sha256"])
    output = args.output_root / f"{args.start}_seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    import torch
    from kaggle_secrets import UserSecretsClient
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              Trainer, TrainerCallback, TrainingArguments, set_seed)
    from torch.utils.data import SequentialSampler
    from flight_run_stats_v0_1 import CausalCollator
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU is required")
    training = config["training"]
    set_seed(args.seed)
    token = UserSecretsClient().get_secret("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT, revision=BASE_REVISION, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_counts = token_accounting(train_rows, tokenizer)
    contract = {
        "start": args.start, "seed": args.seed, "decision_sha256": digest(config),
        "teacher_rows_sha256": file_sha(args.teacher_rows), "training_rows_sha256": digest(train_rows),
        "selection_matrix_sha256": file_sha(REPO / "docs/STATS_CONSOLIDATION_SELECTION_VALIDATION.json"),
        "grader_sha256": file_sha(REPO / "scripts/stats_consolidation_grader.py"),
        "grader_fingerprint": grader_fingerprint(),
        "training_rows": len(train_rows), "new_teacher_rows": len(verified["train"]),
        "historical_replay_rows": 192, "base_revision": BASE_REVISION,
        "v15_baseline_adapter_sha256": file_sha(v15 / "adapter_model.safetensors"),
        "parent_adapter_sha256": file_sha(v15 / "adapter_model.safetensors") if args.start == "original_v15_continued_lora" else None,
        "token_accounting": token_counts,
        "teacher_scope": required_scope,
        "promotion_eligible": not args.pilot_student,
    }
    save_immutable_contract(output / "contract.json", contract)
    save(output / "training_order.json", train_rows)
    completed = read(output / "training_complete.json")
    if completed:
        if not (output / "weight_manifest.json").exists():
            raise RuntimeError("Completion marker exists without verified weight manifest")
        package(output)
        print("RUN ALREADY COMPLETE", json.dumps(completed["selection"])); return
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT, revision=BASE_REVISION, token=token, device_map={"": 0}, torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
    model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.start == "original_v15_continued_lora":
        model = PeftModel.from_pretrained(model, v15, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=training["lora_r"], lora_alpha=training["lora_alpha"], lora_dropout=training["lora_dropout"],
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.config.use_cache = False
    available_updates = math.ceil(len(train_rows) / training["gradient_accumulation_steps"])
    actual_max = min(training["max_updates"], available_updates)
    if len(train_rows) > 390 or available_updates > 49:
        raise RuntimeError("One-pass training exposure exceeds the frozen cap")
    validation_steps = sorted({step for step in (12, 24, actual_max) if step <= actual_max})

    baseline_path = args.output_root / "v15_selection_baseline.json"
    baseline_rows = args.output_root / "v15_selection_rows"
    baseline = read(baseline_path)
    if baseline is None:
        del model; gc.collect(); torch.cuda.empty_cache()
        base_model = AutoModelForCausalLM.from_pretrained(
            STUDENT, revision=BASE_REVISION, token=token, device_map={"": 0}, torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
        base_model = PeftModel.from_pretrained(base_model, v15)
        baseline = {}
        for name, questions in suites.items():
            evaluate(base_model, tokenizer, questions, baseline_rows / (name + ".json"))
            baseline[name] = suite_metrics(read(baseline_rows / (name + ".json")))
        if sum(item["pending"] for item in baseline.values()):
            raise RuntimeError("v15 selection baseline has unresolved semantic scoring")
        save(baseline_path, baseline)
        del base_model; gc.collect(); torch.cuda.empty_cache()
        # Re-enter once so model construction and immutable contract recovery use one path.
        raise RuntimeError("v15 baseline completed; restart this command to begin training")

    existing_history = read(output / "validation_history.json", [])
    existing_pass = next((r for r in existing_history if r["gate"]["passed"]), None)
    if existing_pass:
        adapter = output / f"step_{existing_pass['step']}" / "adapter" / "adapter_model.safetensors"
        if not adapter.exists():
            raise RuntimeError("Passing validation record exists without its portable adapter")
        selection = {"selected_step": existing_pass["step"], "diagnostic_step": None,
                     "stop_reason": "recovered_first_full_validation_pass"}
        save(output / "selection.json", selection)
        save(output / "training_complete.json", {"global_steps": existing_pass["step"], "selection": selection})
        import safetensors.torch
        recovered_weights = safetensors.torch.load_file(str(adapter))
        if not all(torch.isfinite(value).all() for value in recovered_weights.values()):
            raise RuntimeError("Recovered adapter contains non-finite tensors")
        save(output / "weight_manifest.json", {"adapter_sha256": file_sha(adapter),
             "adapter_bytes": adapter.stat().st_size, "finite_tensors": len(recovered_weights),
             "selected_or_diagnostic_step": existing_pass["step"]})
        package(output)
        print("RECOVERED COMPLETE", json.dumps(selection)); return

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
            gate = gate_result(result, baseline, config["validation_gate"])
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
            if state.global_step in validation_steps:
                passed = self.run_validation(state, model)
                control.should_save = True
                if passed or any(r["metrics"][name]["pending"] for r in read(output / "validation_history.json", []) if r["step"] == state.global_step for name in r["metrics"]):
                    control.should_training_stop = True
            return control

    args_train = TrainingArguments(
        output_dir=str(output / "checkpoints"), num_train_epochs=1, max_steps=-1,
        per_device_train_batch_size=1, gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=training["learning_rate"], lr_scheduler_type=training["lr_scheduler_type"],
        warmup_steps=training["warmup_steps"], fp16=True, logging_steps=12,
        save_strategy="steps", save_steps=12, save_total_limit=4, report_to="none", remove_unused_columns=False,
        optim="paged_adamw_8bit", seed=args.seed, data_seed=args.seed, disable_tqdm=True)
    callback = OnlineValidation()
    class OrderedTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

    trainer = OrderedTrainer(model=model, args=args_train, train_dataset=dataset(train_rows, tokenizer),
                      data_collator=CausalCollator(tokenizer), callbacks=[callback])
    checkpoints = sorted((output / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    result = trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)
    history = read(output / "validation_history.json", [])
    passed = next((r for r in history if r["gate"]["passed"]), None)
    pending = any(r["metrics"][name]["pending"] for r in history for name in r["metrics"])
    selection = {"selected_step": passed["step"] if passed else None,
                 "diagnostic_step": None if passed else history[-1]["step"],
                 "stop_reason": "manual_review_required" if pending else ("first_full_validation_pass" if passed else "update_cap_without_full_pass")}
    save(output / "selection.json", selection)
    save(output / "training_complete.json", {"global_steps": trainer.state.global_step,
         "actual_max_updates": actual_max, "available_one_pass_updates": available_updates,
         "training_loss": result.training_loss, "selection": selection, "token_accounting": token_counts,
         "note": "Independent holdout and permanent MC await the four-run frozen evaluator."})
    selected_step = selection["selected_step"] or selection["diagnostic_step"]
    adapter_file = output / f"step_{selected_step}" / "adapter" / "adapter_model.safetensors"
    import safetensors.torch
    weights = safetensors.torch.load_file(str(adapter_file))
    if not all(torch.isfinite(value).all() for value in weights.values()):
        raise RuntimeError("Non-finite adapter tensor")
    save(output / "weight_manifest.json", {"adapter_sha256": file_sha(adapter_file),
         "adapter_bytes": adapter_file.stat().st_size, "finite_tensors": len(weights),
         "selected_or_diagnostic_step": selected_step})
    package(output)
    del trainer, model
    gc.collect(); torch.cuda.empty_cache()
    print("RUN COMPLETE", json.dumps(selection), flush=True)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
