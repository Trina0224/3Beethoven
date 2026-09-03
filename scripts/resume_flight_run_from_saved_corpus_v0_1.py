"""Resume the 3Beethoven statistics flight run from an already-generated teacher corpus.

Use this after teacher generation succeeded but training failed because the installed
Transformers version rejected one or more TrainingArguments keyword names.

This script deliberately DOES NOT regenerate teacher data. It imports the canonical
flight-run module, loads teacher_train_v0_1.jsonl, wraps TrainingArguments with a
version-adaptive compatibility layer, then trains/evaluates/saves the adapter.

No Co-authored-by metadata. No secrets are written to disk.
"""

import inspect
import json
import time
from pathlib import Path

from kaggle_secrets import UserSecretsClient

import flight_run_stats_v0_1 as run


DATA_PATH = Path("/kaggle/working/3beethoven_stats_flight_v0_1/teacher_train_v0_1.jsonl")


def load_records(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Saved teacher corpus not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def install_trainingarguments_compat():
    original = run.TrainingArguments
    params = inspect.signature(original.__init__).parameters

    def compatible_training_arguments(**kwargs):
        # Transformers releases have used both names.
        if "eval_strategy" in kwargs and "eval_strategy" not in params and "evaluation_strategy" in params:
            kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

        # Some releases do not expose warmup_ratio. Use a conservative single warmup step.
        if "warmup_ratio" in kwargs and "warmup_ratio" not in params:
            kwargs.pop("warmup_ratio")
            if "warmup_steps" in params:
                kwargs["warmup_steps"] = 1

        # Filter only genuinely unsupported arguments; print them so nothing is hidden.
        dropped = sorted(k for k in kwargs if k not in params)
        if dropped:
            print("TrainingArguments compatibility: dropping unsupported args:", dropped)
        kwargs = {k: v for k, v in kwargs.items() if k in params}
        return original(**kwargs)

    run.TrainingArguments = compatible_training_arguments
    print("Transformers TrainingArguments compatibility layer installed.")


def main():
    start = time.time()
    records = load_records(DATA_PATH)
    print(f"Recovered teacher examples: {len(records)}")
    if len(records) < 90:
        raise RuntimeError(f"Only {len(records)} saved teacher examples; refusing to train below 90.")

    secrets = UserSecretsClient()
    hf_token = secrets.get_secret("HF_TOKEN")

    install_trainingarguments_compat()

    model, tokenizer, train_result, train_n, eval_n = run.train_adapter(records, hf_token)
    post_acc, post_by_cat, eval_rows = run.evaluate_adapter(model, tokenizer)

    summary = {
        "teacher_model": run.TEACHER_MODEL,
        "student_model": run.STUDENT_MODEL,
        "pipeline_version": "stats-flight-v0.1-resumed",
        "teacher_examples": len(records),
        "train_examples": train_n,
        "validation_examples": eval_n,
        "train_loss": train_result.training_loss,
        "frozen_eval_questions": len(run.FROZEN_EVAL),
        "known_pretrain_targeted_baseline": 0.5625,
        "known_teacher_targeted_score": 1.0,
        "post_distillation_targeted_score": post_acc,
        "post_distillation_by_category": post_by_cat,
        "eval_rows": eval_rows,
        "elapsed_minutes_resume_only": round((time.time() - start) / 60, 1),
        "adapter_dir": str(run.ADAPTER_DIR),
        "recovered_from": str(DATA_PATH),
    }
    run.SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n========================================")
    print("3BEETHOVEN RESUMED FLIGHT RUN COMPLETE")
    print("========================================")
    print(f"Teacher examples recovered: {len(records)}")
    print(f"Training loss: {train_result.training_loss:.4f}")
    print("Targeted baseline before distillation: 56.25%")
    print(f"Targeted score after distillation:    {post_acc:.2%}")
    print("Teacher score on targeted benchmark:  100.00%")
    print(f"By category: {post_by_cat}")
    print(f"Outputs: {run.RUN_DIR}")


if __name__ == "__main__":
    main()
