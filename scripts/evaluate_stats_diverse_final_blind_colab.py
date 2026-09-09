"""Evaluate vanilla 3B and the latest diverse step-180 adapter on final blind.

This is a Colab/Kaggle runtime script. It intentionally compares the original
base model against the latest adapter loaded directly on that same base. It does
not load v15, stack adapters, train, or read development scores for selection.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE = "meta-llama/Llama-3.2-3B-Instruct"
BASE_REVISION = "0cb88a4f764b7a12671c53f0838cd831a0843b95"
LATEST_STEP180_SHA256 = "ff89a22f097e4db457dab287042ae36520ec6b9b36f26e5ed11fe42337c04f23"

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_adapter_dir(path: Path) -> Path:
    """Accept either adapter dir, ZIP extraction root, or downloaded Kaggle output root."""
    path = Path(path)
    if (path / "adapter_model.safetensors").exists():
        return path
    for candidate in (
        path / "adapter",
        path / "student_final2" / "adapter_step_180",
        path / "3beethoven_diverse_run" / "student_final2" / "adapter_step_180",
    ):
        if (candidate / "adapter_model.safetensors").exists():
            return candidate
    matches = sorted(path.rglob("adapter_model.safetensors")) if path.exists() else []
    matches = [p.parent for p in matches if file_sha256(p) == LATEST_STEP180_SHA256]
    if len(matches) != 1:
        raise RuntimeError(f"Could not resolve unique latest step180 adapter under {path}; matches={matches}")
    return matches[0]


def load_final_blind(path: Path):
    payload = read_json(path)
    rows = payload["final_blind"]
    receipt = payload["receipt"]
    if receipt.get("count") != 180 or len(rows) != 180:
        raise RuntimeError("Final blind row count drift")
    category_counts = Counter(row["category"] for row in rows)
    if set(category_counts.values()) != {10} or len(category_counts) != 18:
        raise RuntimeError(f"Unexpected final blind category counts: {category_counts}")
    if receipt.get("split_sha256") != digest(rows):
        raise RuntimeError("Final blind split SHA-256 drift")
    return rows, receipt


def summarize(rows):
    by_category = defaultdict(lambda: {"n": 0, "correct": 0, "pending": 0, "format_strict": 0})
    by_family = defaultdict(lambda: {"n": 0, "correct": 0, "pending": 0})
    failures = []
    for row in rows:
        correct = bool(row["primary_correct"])
        pending = bool(row.get("review_required"))
        item = by_category[row["category"]]
        item["n"] += 1
        item["correct"] += int(correct)
        item["pending"] += int(pending)
        item["format_strict"] += int(bool(row.get("strict_one_line_expression")))
        fam = by_family[row.get("family") or row["question"].split()[0]]
        fam["n"] += 1
        fam["correct"] += int(correct)
        fam["pending"] += int(pending)
        if not correct:
            failures.append({
                "id": row["id"],
                "category": row["category"],
                "family": row.get("family"),
                "raw": row["raw"],
                "reason": row.get("reason"),
                "math_correct": row.get("math_correct"),
                "executable": row.get("executable"),
                "review_required": row.get("review_required"),
                "normalized_expression": row.get("normalized_expression"),
            })
    return {
        "n": len(rows),
        "correct": sum(1 for row in rows if row["primary_correct"]),
        "pending": sum(1 for row in rows if row.get("review_required")),
        "strict_one_line_expression": sum(1 for row in rows if row.get("strict_one_line_expression")),
        "by_category": dict(sorted(by_category.items())),
        "by_family": dict(sorted(by_family.items())),
        "failures": failures,
    }


def evaluate_model(model, tokenizer, questions, out_path: Path, *, max_new_tokens: int, label: str):
    import torch
    from stats_consolidation_grader import grader_fingerprint, score

    cached = read_json(out_path) if out_path.exists() else []
    lookup = {q["id"]: q for q in questions}
    if len({row["id"] for row in cached}) != len(cached):
        raise RuntimeError(f"Duplicate cached IDs in {out_path}")
    for row in cached:
        q = lookup[row["id"]]
        if row["question_sha256"] != digest(q) or row["prompt"] != q["prompt"]:
            raise RuntimeError(f"Cached prompt drift for {row['id']}")
        rescored = score(row["raw"], q)
        if rescored["raw_sha256"] != row["raw_sha256"]:
            raise RuntimeError(f"Cached raw hash drift for {row['id']}")
        if row.get("grader_fingerprint") != grader_fingerprint():
            raise RuntimeError("Grader fingerprint drift; delete cache or regrade intentionally")

    done = {row["id"] for row in cached}
    was_training = model.training
    old_cache = getattr(model.config, "use_cache", None)
    model.eval()
    model.config.use_cache = True
    start = time.time()
    for idx, q in enumerate(questions, 1):
        if q["id"] in done:
            continue
        prompt = q["prompt"]
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(chat, add_special_tokens=False, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )[0][inputs["input_ids"].shape[-1]:]
        raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
        judged = score(raw, q)
        cached.append({
            "id": q["id"],
            "split": q["split"],
            "story_id": q.get("story_id"),
            "contrast_group": q.get("contrast_group"),
            "category": q["category"],
            "family": q.get("family"),
            "question_sha256": digest(q),
            "prompt": prompt,
            "raw": raw,
            "generated_tokens": len(generated),
            "hit_token_limit": len(generated) == max_new_tokens,
            "grader_fingerprint": grader_fingerprint(),
            **judged,
        })
        write_json(out_path, cached)
        if len(cached) % 12 == 0:
            s = summarize(cached)
            print(f"FINAL_BLIND_EVAL {label} {len(cached)}/{len(questions)} correct={s['correct']} pending={s['pending']} elapsed_s={time.time()-start:.1f}", flush=True)
    if old_cache is not None:
        model.config.use_cache = old_cache
    model.train(was_training)
    return cached


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True, help="step180 adapter dir, ZIP extraction root, or Kaggle output root")
    parser.add_argument("--final-blind", type=Path, default=ROOT / "docs" / "STATS_DIVERSE_FINAL_BLIND.json")
    parser.add_argument("--output", type=Path, default=Path("/content/3beethoven_final_blind_eval"))
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="HF token; defaults to HF_TOKEN env var")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--limit", type=int, default=0, help="optional smoke-test row limit; 0 means all 180")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    if not args.hf_token:
        raise RuntimeError("Set HF_TOKEN or pass --hf-token; base model access requires Hugging Face auth")

    questions, receipt = load_final_blind(args.final_blind)
    if args.limit:
        questions = questions[: args.limit]
    adapter = resolve_adapter_dir(args.adapter)
    adapter_sha = file_sha256(adapter / "adapter_model.safetensors")
    if adapter_sha != LATEST_STEP180_SHA256:
        raise RuntimeError(f"Latest adapter SHA mismatch: {adapter_sha}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this 3B final-blind run")
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer_source = adapter if (adapter / "tokenizer_config.json").exists() else BASE
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, revision=None if tokenizer_source != BASE else BASE_REVISION, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = None if args.no_4bit else BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    def load_base():
        set_seed(2027)
        return AutoModelForCausalLM.from_pretrained(
            BASE,
            revision=BASE_REVISION,
            token=args.hf_token,
            device_map={"": 0},
            torch_dtype=torch.float16,
            quantization_config=quant,
        )

    contract = {
        "comparison": "original_3b_base_vs_latest_step180_adapter_on_final_blind",
        "base": BASE,
        "base_revision": BASE_REVISION,
        "latest_step180_adapter_sha256": adapter_sha,
        "adapter_path": str(adapter),
        "final_blind_receipt": receipt,
        "row_count": len(questions),
        "max_new_tokens": args.max_new_tokens,
        "torch_cuda_device": torch.cuda.get_device_name(0),
        "uses_v15_as_baseline": False,
        "adapter_loading_rule": "fresh base 3B + step180 LoRA only; no v15 stacking",
    }
    write_json(args.output / "contract.json", contract)

    print("LOADING original_3b_base", flush=True)
    model = load_base()
    base_rows = evaluate_model(model, tokenizer, questions, args.output / "original_3b_base.json", max_new_tokens=args.max_new_tokens, label="original_3b_base")
    base_summary = summarize(base_rows)
    write_json(args.output / "original_3b_base_summary.json", base_summary)
    print("SUMMARY original_3b_base", json.dumps({k: base_summary[k] for k in ("n", "correct", "pending", "strict_one_line_expression")}, ensure_ascii=False), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("LOADING latest_step180_adapter", flush=True)
    model = PeftModel.from_pretrained(load_base(), adapter, is_trainable=False)
    latest_rows = evaluate_model(model, tokenizer, questions, args.output / "latest_step180_adapter.json", max_new_tokens=args.max_new_tokens, label="latest_step180_adapter")
    latest_summary = summarize(latest_rows)
    write_json(args.output / "latest_step180_adapter_summary.json", latest_summary)
    print("SUMMARY latest_step180_adapter", json.dumps({k: latest_summary[k] for k in ("n", "correct", "pending", "strict_one_line_expression")}, ensure_ascii=False), flush=True)

    base_by_id = {row["id"]: row for row in base_rows}
    latest_by_id = {row["id"]: row for row in latest_rows}
    transitions = Counter(f"{int(base_by_id[q['id']]['primary_correct'])}->{int(latest_by_id[q['id']]['primary_correct'])}" for q in questions)
    paired_losses = [q["id"] for q in questions if base_by_id[q["id"]]["primary_correct"] and not latest_by_id[q["id"]]["primary_correct"]]
    paired_gains = [q["id"] for q in questions if (not base_by_id[q["id"]]["primary_correct"]) and latest_by_id[q["id"]]["primary_correct"]]
    comparison = {
        "contract": contract,
        "original_3b_base": base_summary,
        "latest_step180_adapter": latest_summary,
        "transitions_base_to_latest": dict(sorted(transitions.items())),
        "paired_losses_vs_original_3b_base": paired_losses,
        "paired_gains_vs_original_3b_base": paired_gains,
        "latest_beats_original_by_correct_count": latest_summary["correct"] > base_summary["correct"],
        "latest_has_zero_paired_loss_vs_original_3b_base": len(paired_losses) == 0,
    }
    write_json(args.output / "comparison_summary.json", comparison)
    print("FINAL_BLIND_COMPARISON", json.dumps({
        "original_3b_base_correct": base_summary["correct"],
        "latest_step180_correct": latest_summary["correct"],
        "transitions": dict(sorted(transitions.items())),
        "paired_losses_vs_original_3b_base": len(paired_losses),
        "paired_gains_vs_original_3b_base": len(paired_gains),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

