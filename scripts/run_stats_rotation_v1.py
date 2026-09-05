"""No-training, no-teacher-call option-rotation diagnostic on the frozen holdout."""
import hashlib
import importlib.metadata
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

from flight_run_stats_v0_3 import STUDENT, package, read_json, save_json
from stats_holdout_v1 import questions
from stats_v0_3_common import digest, parse_answer, prompt_for, score_rows

ROOT=Path("/kaggle/working/3beethoven_stats_rotation_v1")


def rotate(r,shift):
    return {**r,"choices":r["choices"][shift:]+r["choices"][:shift],
            "answer_letter":"ABCD"[("ABCD".index(r["answer_letter"])-shift)%4]}


def original_index(pred,shift):
    return ("ABCD".index(pred)+shift)%4 if pred in ("A","B","C","D") else None


def metrics(rows,prior):
    groups={qid:sorted([r for r in rows if r["id"]==qid],key=lambda r:r["shift"])
            for qid in sorted({r["id"] for r in rows})}
    if len(rows)!=240 or len(groups)!=60 or any([r["shift"] for r in g]!=[0,1,2,3] for g in groups.values()):
        raise RuntimeError("Incomplete or duplicated rotation results")
    details=[]
    for qid,g in groups.items():
        choices=[r["original_choice_index"] for r in g]
        details.append({"id":qid,"category":g[0]["category"],"correct_rotations":sum(r["correct"] for r in g),
                        "semantic_consistency":None not in choices and len(set(choices))==1,
                        "constant_letter":len({r["predicted"] for r in g})==1 and g[0]["predicted"]!="INVALID",
                        "predicted_letters":[r["predicted"] for r in g],"original_choice_indices":choices})
    return {"overall":score_rows(rows),"by_shift":{str(k):score_rows([r for r in rows if r["shift"]==k]) for k in range(4)},
            "by_gold_position":{k:score_rows([r for r in rows if r["expected"]==k]) for k in "ABCD"},
            "by_category":{k:score_rows([r for r in rows if r["category"]==k]) for k in sorted({r["category"] for r in rows})},
            "all_four_correct":sum(d["correct_rotations"]==4 for d in details),
            "all_four_wrong":sum(d["correct_rotations"]==0 for d in details),
            "semantically_consistent":sum(d["semantic_consistency"] for d in details),
            "constant_letter_questions":sum(d["constant_letter"] for d in details),
            "correct_rotation_histogram":dict(Counter(d["correct_rotations"] for d in details)),
            "original_order_raw_matches_previous":sum(g[0]["raw"]==prior[qid]["raw"] for qid,g in groups.items()),
            "hit_token_limit":sum(r["hit_token_limit"] for r in rows),"question_details":details}


def main():
    import torch
    from kaggle_secrets import UserSecretsClient
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    candidates=[p for p in Path("/kaggle/input").rglob("adapter_model.safetensors")
                if p.parent.name=="adapter" and (p.parent.parent/"model_environment.json").is_file()]
    if len(candidates)!=1:
        raise RuntimeError("Attach the saved version 5 notebook output; expected one final adapter")
    trained=candidates[0].parent.parent
    holdout=trained.parent/"3beethoven_stats_holdout_v1"
    old_protocol=read_json(holdout/"protocol.json")
    if old_protocol is None:
        raise RuntimeError("Saved holdout protocol is required")
    with candidates[0].open("rb") as handle:
        adapter_sha=hashlib.file_digest(handle,"sha256").hexdigest()
    rows=questions()
    if adapter_sha!=old_protocol["adapter_sha256"] or digest(rows)!=old_protocol["benchmark_sha256"]:
        raise RuntimeError("Adapter or frozen questions differ from prior holdout")
    if torch.cuda.device_count()!=1:
        raise RuntimeError("Exactly one visible GPU required")
    ROOT.mkdir(exist_ok=True)
    protocol={"adapter_sha256":adapter_sha,"benchmark_sha256":digest(rows),
              "student_revision":old_protocol["student_revision"],"questions":60,"shifts":[0,1,2,3],
              "models":["baseline","distilled"],"max_new_tokens":16,"teacher_calls":0,"training":False,
              "primary_metrics":["overall accuracy over four rotations","all-four-correct question count","semantic consistency"],
              "interpretation":"Post-hoc diagnostic on exposed holdout; four cyclic rotations, not all 24 permutations; 60 question clusters, not 240 independent questions"}
    prev=read_json(ROOT/"protocol.json")
    if prev is not None and prev!=protocol:
        raise RuntimeError("Rotation protocol changed; refusing mixed results")
    save_json(ROOT/"protocol.json",protocol)
    save_json(ROOT/"benchmark.json",rows)
    save_json(ROOT/"environment.json",{"gpu":torch.cuda.get_device_name(0),"source_path":str(trained),
              "packages":{p:importlib.metadata.version(p) for p in ("torch","transformers","peft","bitsandbytes")}})
    (ROOT/"source").mkdir(exist_ok=True)
    for name in ("run_stats_rotation_v1.py","stats_holdout_v1.py","stats_v0_3_common.py","flight_run_stats_v0_3.py"):
        shutil.copy2(Path(__file__).with_name(name),ROOT/"source"/name)
    print("ROTATION FROZEN",json.dumps(protocol),flush=True)
    set_seed(226)
    token=UserSecretsClient().get_secret("HF_TOKEN")
    tokenizer=AutoTokenizer.from_pretrained(trained/"adapter")
    model=AutoModelForCausalLM.from_pretrained(STUDENT,revision=old_protocol["student_revision"],token=token,
              device_map={"":0},torch_dtype=torch.float16,
              quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                  bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
    model=prepare_model_for_kbit_training(model,gradient_checkpointing_kwargs={"use_reentrant":False})
    summaries={}
    for name in ("baseline","distilled"):
        if name=="distilled":
            model=PeftModel.from_pretrained(model,trained/"adapter")
        model.eval()
        model.config.use_cache=True
        path=ROOT/(name+".json")
        results=read_json(path,[])
        done={(r["id"],r["shift"]) for r in results}
        for r in rows:
            for shift in range(4):
                if (r["id"],shift) in done:
                    continue
                changed=rotate(r,shift)
                chat=tokenizer.apply_chat_template([{"role":"user","content":prompt_for(changed)}],
                          tokenize=False,add_generation_prompt=True)
                inputs=tokenizer(chat,add_special_tokens=False,return_tensors="pt").to(model.device)
                start=time.monotonic()
                with torch.inference_mode():
                    generated=model.generate(**inputs,max_new_tokens=16,do_sample=False,pad_token_id=tokenizer.eos_token_id)
                output=generated[0][inputs["input_ids"].shape[-1]:]
                raw=tokenizer.decode(output,skip_special_tokens=True).strip()
                pred=parse_answer(raw)
                results.append({"id":r["id"],"category":r["category"],"shift":shift,"expected":changed["answer_letter"],
                      "predicted":pred,"correct":pred==changed["answer_letter"],"raw":raw,
                      "original_choice_index":original_index(pred,shift),"generated_tokens":len(output),
                      "hit_token_limit":len(output)==16,"elapsed_seconds":time.monotonic()-start})
                save_json(path,results)
                if len(results)%40==0:
                    print("ROTATION PROGRESS",name,len(results),"/240",flush=True)
        prior={r["id"]:r for r in read_json(holdout/(name+".json"))}
        summaries[name]=metrics(results,prior)
        print("ROTATION MODEL COMPLETE",name,json.dumps({k:v for k,v in summaries[name].items() if k!="question_details"}),flush=True)
    summary={"protocol":protocol,"models":summaries,"additional_teacher_calls":0,
             "paired_all_four":{"wrong_to_right":sum(a["correct_rotations"]<4 and b["correct_rotations"]==4 for a,b in zip(summaries["baseline"]["question_details"],summaries["distilled"]["question_details"])),
                                "right_to_wrong":sum(a["correct_rotations"]==4 and b["correct_rotations"]<4 for a,b in zip(summaries["baseline"]["question_details"],summaries["distilled"]["question_details"]))}}
    save_json(ROOT/"summary.json",summary)
    package(ROOT)
    archive=ROOT.parent/(ROOT.name+".zip")
    with archive.open("rb") as handle:
        sha=hashlib.file_digest(handle,"sha256").hexdigest()
    print("ROTATION COMPLETE",json.dumps(summary),flush=True)
    print("ROTATION ARCHIVE",archive.stat().st_size,sha,flush=True)


if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    main()
