"""Frozen 60-question, three-model evaluation; no training and <=60 NEW calls."""
import concurrent.futures
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from flight_run_stats_v0_3 import (STUDENT, TEACHER, TeacherClient, package,
                                  read_json, save_json)
from stats_v0_3_common import digest, parse_answer, prompt_for, read_frozen, score_rows
from stats_holdout_v1 import questions, validate

ROOT = Path("/kaggle/working/3beethoven_stats_holdout_v1")
TRAINED = Path("/kaggle/working/3beethoven_stats_flight_v0_3")


def teacher_eval(root, rows, key):
    client=TeacherClient(root,key,60)
    path=root/"teacher.json"
    results=read_json(path,[])
    done={r["id"] for r in results}
    for r in rows:
        if r["id"] in done:
            continue
        started=time.monotonic()
        raw=client.call("holdout_v1_"+r["id"],[{"role":"user","content":prompt_for(r)}],max_tokens=16)
        pred=parse_answer(raw)
        results.append({"id":r["id"],"category":r["category"],"expected":r["answer_letter"],
                        "predicted":pred,"correct":pred==r["answer_letter"],"raw":raw,
                        "elapsed_seconds":time.monotonic()-started})
        save_json(path,results)
        if len(results)%10==0:
            print("HOLDOUT TEACHER",len(results),"/60",flush=True)
    save_json(root/"api_usage.json",client.stats())
    return results


def student_eval(root, rows, key, revision):
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if torch.cuda.device_count()!=1:
        raise RuntimeError("Exactly one visible GPU required")
    baseline=read_json(root/"baseline.json",[])
    distilled=read_json(root/"distilled.json",[])
    if len(baseline)==len(distilled)==60:
        return baseline,distilled
    tokenizer=AutoTokenizer.from_pretrained(TRAINED/"adapter")
    model=AutoModelForCausalLM.from_pretrained(
        STUDENT,revision=revision,token=key,device_map={"":0},torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                       bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
    model=prepare_model_for_kbit_training(model,gradient_checkpointing_kwargs={"use_reentrant":False})
    for name,results in (("baseline",baseline),("distilled",distilled)):
        if name=="distilled":
            model=PeftModel.from_pretrained(model,TRAINED/"adapter")
        model.eval()
        model.config.use_cache=True
        done={x["id"] for x in results}
        for r in rows:
            if r["id"] in done:
                continue
            chat=tokenizer.apply_chat_template([{"role":"user","content":prompt_for(r)}],
                                               tokenize=False,add_generation_prompt=True)
            inputs=tokenizer(chat,add_special_tokens=False,return_tensors="pt").to(model.device)
            started=time.monotonic()
            with torch.inference_mode():
                output=model.generate(**inputs,max_new_tokens=16,do_sample=False,pad_token_id=tokenizer.eos_token_id)
            new=output[0][inputs["input_ids"].shape[-1]:]
            raw=tokenizer.decode(new,skip_special_tokens=True).strip()
            pred=parse_answer(raw)
            results.append({"id":r["id"],"category":r["category"],"expected":r["answer_letter"],
                "predicted":pred,"correct":pred==r["answer_letter"],"raw":raw,
                "generated_tokens":len(new),"hit_token_limit":len(new)==16,
                "elapsed_seconds":time.monotonic()-started})
            save_json(root/(name+".json"),results)
            if len(results)%10==0:
                print("HOLDOUT",name,len(results),"/60",flush=True)
    return baseline,distilled


def summarize(root,rows,baseline,distilled,teacher,protocol):
    assert all(len(x)==60 for x in (baseline,distilled,teacher))
    indexed={name:{r["id"]:r for r in rs} for name,rs in
             (("baseline",baseline),("distilled",distilled),("teacher",teacher))}
    outcomes=[]
    for r in rows:
        a,b,t=(indexed[n][r["id"]] for n in ("baseline","distilled","teacher"))
        outcomes.append({**r,"baseline":a["predicted"],"distilled":b["predicted"],"teacher":t["predicted"],
                         "wrong_to_right":not a["correct"] and b["correct"],
                         "right_to_wrong":a["correct"] and not b["correct"]})
    summary={"protocol":protocol,"scores":{n:score_rows(list(rs.values())) for n,rs in indexed.items()},
             "by_category":{c:{n:score_rows([r for r in rs.values() if r["category"]==c])
                              for n,rs in indexed.items()} for c in sorted({r["category"] for r in rows})},
             "wrong_to_right":sum(r["wrong_to_right"] for r in outcomes),
             "right_to_wrong":sum(r["right_to_wrong"] for r in outcomes),
             "api_usage":read_json(root/"api_usage.json"),
             "limits":["One seed and 60 internally authored questions; no independent external benchmark",
                       "Authored after training and before inference; none may be used as training targets",
                       "New transfer tasks are harder/different from the old development set; compare models within this set",
                       "Exact overlap checks do not exclude semantic overlap",
                       "No training or prompt tuning after observing these results"]}
    save_json(root/"summary.json",summary)
    save_json(root/"question_outcomes.json",outcomes)
    return summary


def main():
    ROOT.mkdir(exist_ok=True)
    rows=questions()
    old_train=read_json(TRAINED/"curriculum.json")
    old_exam=read_json(TRAINED/"development_benchmark.json")
    if old_train is None or old_exam is None:
        raise RuntimeError("Prior preserved run is required; do not regenerate training")
    audited=validate(old_train+old_exam)
    env=read_json(TRAINED/"model_environment.json")
    adapter=TRAINED/"adapter/adapter_model.safetensors"
    with adapter.open("rb") as handle:
        adapter_sha=hashlib.file_digest(handle,"sha256").hexdigest()
    protocol={"benchmark_sha256":digest(rows),"adapter_sha256":adapter_sha,
              "student_revision":env["student_revision"],"teacher":TEACHER,"student":STUDENT,
              "new_teacher_attempt_cap":60,"prior_experiment_attempts":94,"max_new_tokens":16,
              "prompt":"unchanged v0.3 primary letter-only prompt","training":False,
              "benchmark_status":"post-training internally authored holdout frozen before inference"}
    previous=read_json(ROOT/"protocol.json")
    if previous and previous!=protocol:
        raise RuntimeError("Frozen protocol changed; refusing mixed results")
    save_json(ROOT/"protocol.json",protocol)
    save_json(ROOT/"benchmark.json",rows)
    save_json(ROOT/"benchmark_audit.json",audited)
    (ROOT/"source").mkdir(exist_ok=True)
    for name in ("run_stats_holdout_v1.py","stats_holdout_v1.py","flight_run_stats_v0_3.py","stats_v0_3_common.py"):
        shutil.copy2(Path(__file__).with_name(name),ROOT/"source"/name)
    print("HOLDOUT FROZEN",json.dumps(protocol),flush=True)
    from kaggle_secrets import UserSecretsClient
    secrets=UserSecretsClient()
    teacher_key=secrets.get_secret("OPENROUTER_API_KEY")
    student_key=secrets.get_secret("HF_TOKEN")
    # One serial network worker, separate ledger; GPU work overlaps network wait.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(teacher_eval,ROOT,rows,teacher_key)
        baseline,distilled=student_eval(ROOT,rows,student_key,env["student_revision"])
        teacher=future.result()
    summary=summarize(ROOT,rows,baseline,distilled,teacher,protocol)
    package(ROOT)
    archive=ROOT.parent/(ROOT.name+".zip")
    with archive.open("rb") as handle:
        archive_sha=hashlib.file_digest(handle,"sha256").hexdigest()
    print("HOLDOUT COMPLETE",json.dumps(summary),flush=True)
    print("HOLDOUT ARCHIVE",archive.stat().st_size,archive_sha,flush=True)


if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    main()
