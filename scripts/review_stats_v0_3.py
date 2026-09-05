"""Teacher-only revisions from a pre-training content audit; no extra call budget."""
import json
import os
import shutil
from pathlib import Path

from flight_run_stats_v0_3 import TeacherClient, package, read_jsonl, save_json
from stats_v0_3_common import audit, digest, parse_teacher, prompt_for, read_frozen

REVIEW = {
    "poisson_mean_variance_07": "Avoid claiming that equal mean and variance uniquely identifies the Poisson distribution; it is a property, not a characterization of all distributions.",
    "linear_expectation_08": "The earlier common_mistake incorrectly called E[2X]-E[15] invalid. That identity is valid. Describe a genuinely wrong calculation, such as omitting the subtraction of the constant.",
    "confidence_level_interval_width_01": "The earlier explanation incorrectly equated lower confidence with less precision. A narrower interval is more precise, but has lower coverage. Explain the smaller normal critical value and margin of error; do not confuse coverage with precision.",
    "confidence_level_interval_width_05": "The earlier explanation incorrectly equated lower confidence with less precision. A narrower interval is more precise, but has lower coverage. Explain the smaller normal critical value and margin of error; do not confuse coverage with precision.",
    "confidence_level_interval_width_07": "The earlier explanation incorrectly equated lower confidence with less precision. A narrower interval is more precise, but has lower coverage. Explain the smaller normal critical value and margin of error; do not confuse coverage with precision.",
}


def main():
    from kaggle_secrets import UserSecretsClient
    root = Path("/kaggle/working/3beethoven_stats_flight_v0_3")
    if (root / "training_complete.json").exists() or (root / "split.json").exists():
        raise RuntimeError("Content revisions must occur before training")
    path = root / "teacher_train.jsonl"
    original = root / "teacher_train_before_content_review.jsonl"
    if not original.exists():
        shutil.copy2(path, original)
    records = read_jsonl(original)
    client = TeacherClient(root, UserSecretsClient().get_secret("OPENROUTER_API_KEY"), 120)
    for item in records:
        item.setdefault("reference_conditioned", False)
        if item["id"] not in REVIEW:
            continue
        question = prompt_for(item, "explain").split("\n\nChoose A, B, C, or D")[0]
        messages = [
            {"role": "system", "content": "Return JSON with string fields answer_letter, explanation, common_mistake. Give a concise, mathematically accurate explanation. Identify one genuinely incorrect misconception explicitly as incorrect. Do not describe valid identities as mistakes."},
            {"role": "user", "content": question + "\n\nReference check: " + item["reference_reason"] + " Correct choice: " + item["answer_letter"] + ".\nContent review: " + REVIEW[item["id"]]},
        ]
        obj = parse_teacher(client.call("content_review_" + item["id"] + "_0", messages, json_mode=True))
        if obj.get("answer_letter") != item["answer_letter"]:
            raise RuntimeError("Revised answer disagrees with reference: " + item["id"])
        if any(not isinstance(obj.get(k), str) or len(obj[k]) < n for k, n in (("explanation", 60), ("common_mistake", 15))):
            raise RuntimeError("Invalid revised explanation schema")
        item.update(explanation=obj["explanation"], common_mistake=obj["common_mistake"],
                    reference_conditioned=True, content_review_revision=True)
        print("REVISED", json.dumps(item, ensure_ascii=False), flush=True)
    benchmark = read_frozen(Path(__file__).with_name("flight_run_stats_v0_2.py"))
    report = audit(records, benchmark)
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    save_json(root / "data_audit.json", report)
    save_json(root / "content_review.json", {"reviewed_questions": 60, "revised_ids": list(REVIEW),
              "feedback": REVIEW, "before_sha256": digest(read_jsonl(original)), "after_sha256": digest(records),
              "reference_conditioned_ids": [r["id"] for r in records if r["reference_conditioned"]],
              "api_usage": client.stats(), "target_source": "Llama teacher responses only; audit feedback is not a student target"})
    shutil.copy2(__file__, root / "source" / Path(__file__).name)
    package(root)
    print("CONTENT REVIEW COMPLETE", client.stats(), flush=True)


if __name__ == "__main__":
    main()
