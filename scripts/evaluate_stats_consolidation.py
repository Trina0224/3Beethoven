"""Frozen final evaluator for base, v15, and four selected/diagnostic runs."""
import argparse
import gc
import json
import os
from collections import defaultdict
from pathlib import Path

from flight_run_stats_v0_3 import STUDENT, package, read_json, save_json
from run_stats_consolidation_compare import file_sha, locate_v15, suite_metrics
from run_stats_v0_4 import BASE_REVISION, evaluate as evaluate_mc
from stats_consolidation_eval import evaluate
from stats_consolidation_pilot import digest, assert_execution_released

REPO = Path(__file__).resolve().parents[1]


def promotion(candidate, baseline, rule):
    checks = {"no_pending": sum(candidate[k]["pending"] for k in ("old", "chain", "event")) == 0}
    for suite in ("old", "chain"):
        srule = rule[suite]
        checks[suite + ":total"] = candidate[suite]["correct"] >= baseline[suite]["correct"] - srule["allowed_total_loss"]
        for category, base in baseline[suite]["category_counts"].items():
            checks[f"{suite}:{category}"] = candidate[suite]["category_counts"][category]["correct"] >= base["correct"] - srule["allowed_per_category_loss"]
    erule = rule["event"]
    checks["event:total_absolute"] = candidate["event"]["correct"] >= erule["minimum_correct_total"]
    checks["event:total_gain"] = candidate["event"]["correct"] >= baseline["event"]["correct"] + erule["minimum_gain_total"]
    ce = candidate["event"]["category_counts"]["exactly_one"]["correct"]
    be = baseline["event"]["category_counts"]["exactly_one"]["correct"]
    checks["event:exactly_one_absolute"] = ce >= erule["exactly_one_minimum_correct"]
    checks["event:exactly_one_gain"] = ce >= be + erule["exactly_one_minimum_gain"]
    checks["mc:retention"] = candidate["mc"]["correct"] >= baseline["mc"]["correct"] - rule["mc"]["allowed_total_loss"]
    headroom = {"event_total_gain_attainable": baseline["event"]["correct"] + erule["minimum_gain_total"] <= erule["n"],
                "exactly_one_gain_attainable": be + erule["exactly_one_minimum_gain"] <= erule["exactly_one_n"]}
    return {"passed": all(checks.values()), "checks": checks, "headroom": headroom,
            "unattainable_predeclared_gain": not all(headroom.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working/3beethoven_stats_consolidation"))
    parser.add_argument("--v15-root", type=Path, default=Path("/kaggle/input"))
    args = parser.parse_args()
    decision = read_json(REPO / "docs/STATS_CONSOLIDATION_DECISION_DRAFT.json")
    assert_execution_released(decision)
    if decision.get("protocol_status") != "frozen_for_execution" or decision["promotion_gate"].get("status") != "frozen_for_execution":
        raise RuntimeError("Final evaluation protocol is not frozen")
    holdout = read_json(REPO / "docs/STATS_CONSOLIDATION_HOLDOUT.json")
    anchor = read_json(REPO / "docs/STATS_PERMANENT_ANCHOR_V1.json")
    v15 = locate_v15(args.v15_root, decision["training"]["v15_adapter_sha256"])
    candidates = {"base": None, "v15": v15}
    validation_pass = {}
    for start in decision["training"]["starts"]:
        for seed in decision["training"]["seeds"]:
            run = args.output_root / f"{start}_seed_{seed}"
            selection = read_json(run / "selection.json")
            if not selection:
                raise RuntimeError("Missing run selection: " + str(run))
            step = selection.get("selected_step") or selection.get("diagnostic_step")
            label = f"{start}_seed_{seed}"
            candidates[label] = run / f"step_{step}" / "adapter"
            validation_pass[label] = selection.get("selected_step") is not None

    import torch
    from kaggle_secrets import UserSecretsClient
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    token = UserSecretsClient().get_secret("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(v15)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = read_json(args.output_root / "final_metrics.json", {})
    for label, adapter in candidates.items():
        if label in results:
            continue
        model = AutoModelForCausalLM.from_pretrained(
            STUDENT, revision=BASE_REVISION, token=token, device_map={"": 0}, torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
        if adapter is not None:
            model = PeftModel.from_pretrained(model, adapter)
        folder = args.output_root / "final_eval" / label
        metrics = {}
        for suite, questions in holdout["suites"].items():
            evaluate(model, tokenizer, questions, folder / (suite + ".json"))
            metrics[suite] = suite_metrics(read_json(folder / (suite + ".json")))
        mc = evaluate_mc(model, tokenizer, anchor["mc"]["questions"], folder / "mc.json")
        metrics["mc"] = {"n": mc["overall"]["n"], "correct": mc["overall"]["correct"]}
        metrics["model"] = {"adapter_sha256": file_sha(Path(adapter) / "adapter_model.safetensors") if adapter else None,
                            "base_revision": BASE_REVISION}
        results[label] = metrics
        save_json(args.output_root / "final_metrics.json", results)
        del model; gc.collect(); torch.cuda.empty_cache()

    baseline = results["v15"]
    gates = {label: promotion(metrics, baseline, decision["promotion_gate"])
             for label, metrics in results.items() if label not in ("base", "v15")}
    by_start = defaultdict(list)
    for label, gate in gates.items():
        start = label.rsplit("_seed_", 1)[0]
        by_start[start].append(gate["passed"] and validation_pass[label])
    summary = {"holdout_sha256": digest(holdout), "mc_sha256": digest(anchor["mc"]["questions"]),
               "validation_pass": validation_pass, "promotion_by_run": gates,
               "promotion_by_start": {start: len(values) == 2 and all(values) for start, values in by_start.items()},
               "rule": "both fixed seeds must pass selection and frozen final promotion gates"}
    save_json(args.output_root / "final_summary.json", summary)
    package(args.output_root)
    print(json.dumps(summary))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
