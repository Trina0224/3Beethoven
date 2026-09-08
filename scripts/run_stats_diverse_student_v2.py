"""Run the single frozen diverse-curriculum student trajectory on Kaggle."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import glob
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from flight_run_stats_v0_1 import CausalCollator
from run_stats_v0_4 import BASE_REVISION, dataset
from stats_diverse_training_contract import digest, interleave_training_rows

BASE = "meta-llama/Llama-3.2-3B-Instruct"
PARENT_SHA256 = "9369d52de4a886df9da0c872cd41bd4e01af0a38bf02ad724b5951c1a6b9f5d3"
ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_training_rows(curriculum, replay, teacher):
    if teacher.get("training_release") is not True:
        raise RuntimeError("Teacher results are not released")
    results = {row["training_row_id"]: row for row in teacher["rows"]}
    if len(results) != 720:
        raise RuntimeError("Teacher result identity/count drift")
    new_rows = []
    for source in curriculum["train"]:
        result = results[source["id"]]
        if result["status"] not in {"raw_teacher_accepted", "canonical_corrected"}:
            raise RuntimeError("Unusable teacher target")
        if not all(result["training_target_gates"].values()):
            raise RuntimeError("Teacher target gate failed")
        row = dict(source)
        row["target"] = result["training_target"]
        row["target_provenance"] = result["target_provenance"]
        new_rows.append(row)
    ordered = interleave_training_rows(new_rows, replay["replay"])
    rows = [item["row"] for item in ordered]
    return rows, {
        "rows": len(rows),
        "new_rows": 720,
        "replay_rows": 720,
        "ordered_ids_sha256": digest([(item["training_role"], item["row"]["id"]) for item in ordered]),
        "teacher_results_sha256": sha(args.teacher),
    }


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Output exists; refusing a second student trajectory")
    if not (args.parent / "adapter_model.safetensors").exists():
        matches = [Path(path).parent for path in glob.glob(
            "/kaggle/input/**/v15_probe/adapter/adapter_model.safetensors", recursive=True
        ) if sha(path) == PARENT_SHA256]
        if not matches:
            import kagglehub
            saved = Path(kagglehub.notebook_output_download(
                "trinashih/3beethoven-v0-2/versions/29"
            ))
            matches = [path.parent for path in saved.rglob("adapter_model.safetensors")
                       if sha(path) == PARENT_SHA256]
        if len(matches) != 1:
            raise RuntimeError(f"Could not resolve the frozen v15 parent: {matches}")
        args.parent = matches[0]
    if not args.teacher.exists():
        matches = [Path(path) for path in glob.glob(
            "/kaggle/input/**/STATS_DIVERSE_TEACHER_RESULTS.json", recursive=True
        )]
        if len(matches) != 1:
            raise RuntimeError(f"Could not resolve frozen teacher results: {matches}")
        args.teacher = matches[0]
    if sha(args.parent / "adapter_model.safetensors") != PARENT_SHA256:
        raise RuntimeError("v15 parent SHA-256 mismatch")

    curriculum = read(ROOT / "docs/STATS_DIVERSE_CURRICULUM.json")
    replay = read(ROOT / "docs/STATS_DIVERSE_REPLAY.json")
    teacher = read(args.teacher)
    rows, receipt = build_training_rows(curriculum, replay, teacher)
    args.output.mkdir(parents=True)
    write(args.output / "training_plan_receipt.json", receipt)

    import torch
    from kaggle_secrets import UserSecretsClient
    from peft import PeftModel, prepare_model_for_kbit_training
    from safetensors.torch import load_file
    from torch.utils.data import SequentialSampler
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed

    if torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA GPU is required")
    set_seed(2027)
    token = UserSecretsClient().get_secret("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prepared = dataset(rows, tokenizer)
    if len(prepared) != 1440:
        raise RuntimeError("Prepared row count drift")
    model = AutoModelForCausalLM.from_pretrained(
        BASE, revision=BASE_REVISION, token=token, device_map={"": 0}, torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
    )
    model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(model, args.parent, is_trainable=True)
    model.config.use_cache = False

    class OrderedTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

        def training_step(self, model, inputs, *a, **kw):
            item = {
                "global_step_before": self.state.global_step,
                "input_ids_sha256": digest(inputs["input_ids"].detach().cpu().tolist()),
                "labels_sha256": digest(inputs["labels"].detach().cpu().tolist()),
                "supervised_tokens": int((inputs["labels"] != -100).sum().item()),
            }
            loss = super().training_step(model, inputs, *a, **kw)
            with (args.output / "microbatches.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(item) + "\n")
            return loss

    env = {name: importlib.metadata.version(name) for name in
           ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "datasets")}
    write(args.output / "environment.json", {"gpu": torch.cuda.get_device_name(0), "packages": env})
    training_args = TrainingArguments(
        output_dir=str(args.output / "checkpoints"), num_train_epochs=1,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=1e-5, lr_scheduler_type="constant", warmup_steps=0,
        fp16=True, logging_steps=10, save_strategy="steps", save_steps=60,
        save_total_limit=3, report_to="none", remove_unused_columns=False,
        optim="paged_adamw_8bit", seed=2027, data_seed=2027, disable_tqdm=True,
    )
    trainer = OrderedTrainer(model=model, args=training_args, train_dataset=prepared,
                             data_collator=CausalCollator(tokenizer))
    result = trainer.train()
    if trainer.state.global_step != 180:
        raise RuntimeError("Optimizer update count drift")
    model.save_pretrained(args.output / "adapter_step_180")
    tokenizer.save_pretrained(args.output / "adapter_step_180")
    tensors = load_file(args.output / "adapter_step_180" / "adapter_model.safetensors")
    if not all(torch.isfinite(tensor).all() for tensor in tensors.values()):
        raise RuntimeError("Non-finite adapter tensor")
    trace = [json.loads(line) for line in (args.output / "microbatches.jsonl").read_text().splitlines()]
    if len(trace) != 1440:
        raise RuntimeError("Training trace count drift")
    write(args.output / "training_complete.json", {
        "updates": 180, "rows": 1440, "loss": result.training_loss,
        "adapter_sha256": sha(args.output / "adapter_step_180" / "adapter_model.safetensors"),
        "checkpoints": [60, 120, 180], "trace_rows": len(trace),
        "teacher_results_sha256": sha(args.teacher), "parent_sha256": PARENT_SHA256,
    })
    print("DIVERSE_STUDENT_TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
