"""Historical eight-family transfer diagnostic; never used for model selection."""
import gc,json,shutil,os
from pathlib import Path
from run_stats_v0_18 import ROOT
from run_stats_v0_17 import P15,HASHES,sha,evaluate
from stats_curriculum_v0_17 import build,digest,prompt,score
from flight_run_stats_v0_3 import save_json as save,read_json as read,STUDENT

def main():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,set_seed
    from peft import PeftModel
    from kaggle_secrets import UserSecretsClient
    from run_stats_v0_4 import BASE_REVISION
    root=ROOT/'transfer';root.mkdir(exist_ok=True);qs=build()['test']
    assert sha(P15/'adapter/adapter_model.safetensors')==HASHES[15]
    stages=read(ROOT/'stage_summary.json');n=len(stages)
    old=ROOT.parent/'3beethoven_stats_v0_17/v15_test.json'
    if not (root/'v15.json').exists():shutil.copy2(old,root/'v15.json')
    lookup={q['id']:q for q in qs};baseline=read(root/'v15.json')
    assert len(baseline)==96 and {r['id'] for r in baseline}==set(lookup)
    for r in baseline:
        q=lookup[r['id']];assert r['question_sha256']==digest(q) and r['prompt']==prompt(q)
        assert all(r[k]==v for k,v in score(r['raw'],q).items())
    tok=AutoTokenizer.from_pretrained(P15/'adapter');token=UserSecretsClient().get_secret('HF_TOKEN')
    for name in (f'stage_{n}',f'control_{n}'):
        if len(read(root/(name+'.json'),[]))==96:continue
        set_seed(1818)
        base=AutoModelForCausalLM.from_pretrained(STUDENT,revision=BASE_REVISION,token=token,device_map={'':0},torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
        model=PeftModel.from_pretrained(base,ROOT/name/'adapter')
        m=evaluate(model,tok,qs,root/(name+'.json'));print('V18 HISTORICAL TRANSFER',name,m,flush=True)
        del model,base;gc.collect();torch.cuda.empty_cache()
    outputs={p.stem:read(p) for p in root.glob('*.json')};metrics={}
    for name,rows in outputs.items():
        assert len(rows)==96 and {r['id'] for r in rows}==set(lookup)
        for r in rows:
            q=lookup[r['id']];assert r['question_sha256']==digest(q) and r['prompt']==prompt(q)
            assert all(r[k]==v for k,v in score(r['raw'],q).items())
        metrics[name]=dict(correct=sum(r['correct'] for r in rows),pending=sum(r['review_required'] for r in rows),
            by_category={c:sum(r['correct'] for r in rows if r['category']==c) for c in sorted({q['category'] for q in qs})})
    result=dict(protocol='Secondary historical v17 eight-family diagnostic registered during control training, before any v18 test results. Never selects a checkpoint. v15 responses reused only after exact prompt/question/hash/score validation; all candidate responses freshly generated.',
                questions=qs,outputs=outputs,metrics=metrics,reused_responses=96,new_responses=192)
    save(ROOT/'transfer_results.json',result)

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0';main()
