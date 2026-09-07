"""Read-only fixed-weight controls before repairing first-student regression."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse, json, hashlib, gc, sys, importlib.metadata as md
from pathlib import Path

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x): Path(p).write_text(json.dumps(x,indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    sys.path.insert(0,str(a.repo/'scripts'))
    from stats_consolidation_eval import evaluate
    from run_stats_consolidation_compare import suite_metrics
    from stats_curriculum_v0_18 import prompt
    from stats_consolidation_grader import grader_fingerprint
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    from peft import PeftModel,prepare_model_for_kbit_training
    from kaggle_secrets import UserSecretsClient
    a.output.mkdir(exist_ok=True,parents=True)
    root=a.input/'3beethoven_first_students';student=root/'original_v15_continued_lora_seed_31415'
    contract=read(student/'contract.json')
    assert grader_fingerprint()==contract['grader_fingerprint']
    env={n:md.version(n) for n in ['torch','transformers','peft','bitsandbytes','accelerate','datasets']}
    save(a.output/'environment.json',env);print('ENV',env,flush=True)
    assert env==read(root/'environment.json'),(env,read(root/'environment.json'))
    suites=read(a.repo/'docs/STATS_CONSOLIDATION_SELECTION_VALIDATION.json')['suites']
    lookup={q['id']:q for v in (14,15) for q in read(a.repo/f'docs/STATS_V0_{v}_FROZEN_QUESTIONS.json')['train']}
    replay=[r for r in read(a.repo/'docs/STATS_V0_19_REPLAY_SOURCE.json') if r['category'] in ('interval','poisson_time','uniform_time')]
    suites['replay']=[]
    for row in replay:
        q=lookup[row['source_id']];assert prompt(q)==row['prompt'];suites['replay'].append(q)
    assert len(replay)==72
    save(a.output/'diagnostic_questions.json',suites)
    parent=a.input/'first_batch_parent/3beethoven_stats_v0_15/adapter'
    assert sha(parent/'adapter_model.safetensors')==contract['parent_adapter_sha256']
    token=UserSecretsClient().get_secret('HF_TOKEN')
    modelid='meta-llama/Llama-3.2-3B-Instruct'
    tokenizer=AutoTokenizer.from_pretrained(modelid,revision=contract['base_revision'],token=token)
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    all_results={}
    for label,adapter,prepared,oldfolder in [('v15',parent,False,root/'v15_selection_rows'),('step24',student/'step_24/adapter',True,student/'step_24'),('step33',student/'step_33/adapter',True,student/'step_33')]:
        if prepared:assert sha(adapter/'adapter_model.safetensors')==read(adapter.parent/'adapter_hash.json')['sha256']
        print('LOADING',label,sha(adapter/'adapter_model.safetensors'),flush=True)
        base=AutoModelForCausalLM.from_pretrained(modelid,revision=contract['base_revision'],token=token,device_map={'':0},torch_dtype=torch.float16,quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
        if prepared:base=prepare_model_for_kbit_training(base,gradient_checkpointing_kwargs={'use_reentrant':False})
        model=PeftModel.from_pretrained(base,adapter,is_trainable=prepared)
        folder=a.output/label;folder.mkdir(exist_ok=True)
        all_results[label]={'adapter_sha256':sha(adapter/'adapter_model.safetensors'),'prepared':prepared,'suites':{}}
        for name,questions in suites.items():
            rows=evaluate(model,tokenizer,questions,folder/f'{name}.json')
            m=suite_metrics(rows)
            if name!='replay':
                prev={r['id']:r for r in read(oldfolder/f'{name}.json')}
                m['same_raw']=sum(r['raw']==prev[r['id']]['raw'] for r in rows)
                m['raw_changes']=[{'id':r['id'],'before':prev[r['id']]['raw'],'after':r['raw']} for r in rows if r['raw']!=prev[r['id']]['raw']]
            all_results[label]['suites'][name]=m;save(a.output/'CONTROL_RESULTS.json',all_results)
            print('CONTROL',label,name,json.dumps(m),flush=True)
        del model,base;gc.collect();torch.cuda.empty_cache()
    save(a.output/'CONTROL_COMPLETE.json',{'complete':True,'models':list(all_results),'teacher_calls':0,'training_updates':0})
    print('CONTROL_COMPLETE',flush=True)
if __name__=='__main__':main()
