"""Resumable greedy evaluator using the isolated consolidation grader."""
from flight_run_stats_v0_3 import read_json, save_json
from stats_consolidation_grader import score
from stats_curriculum_v0_18 import prompt
from stats_consolidation_pilot import digest


def evaluate(model, tokenizer, questions, path, max_new_tokens=160):
    import torch
    rows = read_json(path, [])
    lookup = {q["id"]: q for q in questions}
    if len({r["id"] for r in rows}) != len(rows):
        raise RuntimeError("Duplicate evaluation IDs")
    for row in rows:
        q = lookup[row["id"]]
        if row["question_sha256"] != digest(q) or row["prompt"] != prompt(q):
            raise RuntimeError("Cached evaluation input drift")
        rescored = score(row["raw"], q)
        if row["raw_sha256"] != rescored["raw_sha256"]:
            raise RuntimeError("Cached raw response hash drift")
    done = {r["id"] for r in rows}
    was_training, old_cache = model.training, model.config.use_cache
    model.eval(); model.config.use_cache = True
    for q in questions:
        if q["id"] in done:
            continue
        user_prompt = prompt(q)
        chat = tokenizer.apply_chat_template([{"role": "user", "content": user_prompt}],
                                             tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat, add_special_tokens=False, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                       pad_token_id=tokenizer.eos_token_id)[0][inputs["input_ids"].shape[-1]:]
        raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
        rows.append({"id": q["id"], "story_id": q.get("story_id"), "category": q["category"],
                     "depth": q.get("depth"), "question_sha256": digest(q), "prompt": user_prompt,
                     "raw": raw, "generated_tokens": len(generated),
                     "hit_token_limit": len(generated) == max_new_tokens, **score(raw, q)})
        save_json(path, rows)
    model.config.use_cache = old_cache
    model.train(was_training)
    return rows

