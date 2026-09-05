"""Reload the saved adapter and verify the packaged run without teacher calls."""
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path


def main():
    import torch
    from kaggle_secrets import UserSecretsClient
    from peft import PeftModel, prepare_model_for_kbit_training
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from flight_run_stats_v0_3 import STUDENT, package, read_json, save_json
    from stats_v0_3_common import parse_answer, prompt_for
    root = Path("/kaggle/working/3beethoven_stats_flight_v0_3")
    if not (root / "summary.json").exists():
        raise RuntimeError("Full evaluation must finish before final verification")
    adapter = root / "adapter"
    with safe_open(str(adapter / "adapter_model.safetensors"), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if not keys or not all("lora_" in k for k in keys):
            raise RuntimeError("Unexpected adapter tensor inventory")
        for key in keys:
            if not torch.isfinite(handle.get_tensor(key)).all().item():
                raise RuntimeError("Nonfinite adapter tensor")
    token = UserSecretsClient().get_secret("HF_TOKEN")
    env = read_json(root / "model_environment.json")
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT, revision=env["student_revision"], token=token, device_map={"": 0}, torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                              bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
    model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    model.config.use_cache = True
    r = read_json(root / "development_benchmark.json")[0]
    chat = tokenizer.apply_chat_template([{"role": "user", "content": prompt_for(r)}],
                                         tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        result = model.generate(**inputs, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    raw = tokenizer.decode(result[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    stored = next(x for x in read_json(root / "post_eval.json") if x["id"] == r["id"] and x["mode"] == "letter16")
    if raw != stored["raw"]:
        raise RuntimeError("Reloaded adapter differs on the smoke-test response")
    verification = {"finite_adapter_tensors": len(keys), "reloaded_adapter": True,
                    "smoke_question_id": r["id"], "raw": raw, "predicted": parse_answer(raw),
                    "matches_stored_raw": True, "teacher_calls_in_verification": 0,
                    "student_revision": env["student_revision"]}
    save_json(root / "artifact_verification.json", verification)
    shutil.copy2(__file__, root / "source" / Path(__file__).name)
    package(root)
    archive = root.parent / (root.name + ".zip")
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read("manifest.json"))
        for entry in manifest:
            content = z.read(entry["path"])
            if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise RuntimeError("ZIP manifest mismatch: " + entry["path"])
    with archive.open("rb") as handle:
        archive_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    print("FINAL ARTIFACT VERIFIED", json.dumps({**verification, "manifest_files_verified": len(manifest),
          "archive_bytes": archive.stat().st_size, "archive_sha256": archive_sha}), flush=True)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
