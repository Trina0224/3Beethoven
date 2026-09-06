"""Paired post-test diagnostic; no training, teacher calls or headline rescoring."""
import gc
import hashlib
import json
from pathlib import Path
from exact_calculator import calculate
from formulation_grader import grade
from run_stats_v0_15 import save, read

ROOT = Path('/kaggle/working/3beethoven_stats_v0_15_diagnostic')
HASHES = {'v14':'c7def77757fefaaf41db6938500159795a47503dac54d72d79113de47a3239a5',
          'v15':'9369d52de4a886df9da0c872cd41bd4e01af0a38bf02ad724b5951c1a6b9f5d3'}

def build():
    source=json.loads((Path(__file__).resolve().parents[1]/'docs/STATS_V0_15_FROZEN_QUESTIONS.json').read_text())
    rows=[]
    for q in source['test']:
        if q['category'] not in ('moment','poisson_scaled'): continue
        b=q['bindings'];m,a,z=(b[k] for k in ('mean','scale','offset'))
        v=b.get('variance',m)
        # Keep the original story but remove its final request when asking a subtask.
        story=q['question'].split(' Set up that expectation.')[0].split(' What variance')[0]
        tasks=[('extract_mean','Return only the numerical value of E[X].',m),
               ('extract_variance','Return only the numerical value of Var(X).',v),
               ('extract_scale','Return only the multiplier applied to X.',a),
               ('extract_offset','Return only the constant added after multiplying X.',z),
               ('mean','Set up E[Y], where Y is the transformed quantity before any squaring.',f'{a}*{m}+{z}'),
               ('variance','Set up Var(Y), where Y is the transformed quantity before any squaring.',f'{a}**2*{v}'),
               ('original',q['question'],q['expression']),
               ('hinted','Use Var(a*X+b)=a**2*Var(X) and E[Y**2]=Var(Y)+E[Y]**2. '+q['question'],q['expression'])]
        for stage,request,expr in tasks:
            p=request if stage in ('original','hinted') else story+'\n'+request
            if not stage.startswith('extract_'):
                p+='\nReturn only Expression: followed by a fully substituted numerical expression. Keep arithmetic unevaluated.'
            rows.append(dict(id=q['id']+'_'+stage,story_id=q['id'],family=q['category'],stage=stage,
                prompt=p,category='diagnostic',bindings={},expression=expr,answer=calculate(expr)))
    return rows

def main():
    import torch
    from kaggle_secrets import UserSecretsClient
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    from peft import PeftModel
    from flight_run_stats_v0_3 import STUDENT
    from run_stats_v0_4 import BASE_REVISION
    ROOT.mkdir(exist_ok=True)
    questions=build();save(ROOT/'questions.json',questions)
    token=UserSecretsClient().get_secret('HF_TOKEN')
    adapters={}
    for name in HASHES:
        folder=Path('/kaggle/working/3beethoven_stats_v0_'+name[1:])
        target=folder/'adapter/adapter_model.safetensors'
        if not target.exists():
            import kagglehub,shutil
            cached=Path(kagglehub.notebook_output_download('trinashih/3beethoven-v0-2/versions/29'))
            matches=[p for p in cached.rglob('adapter_model.safetensors') if p.parent.parent.name==folder.name]
            assert matches, name+' saved adapter missing'
            shutil.copytree(matches[0].parent,folder/'adapter',dirs_exist_ok=True)
        assert hashlib.sha256(target.read_bytes()).hexdigest()==HASHES[name]
        adapters[name]=folder/'adapter'
    all_results={}
    for name,adapter in adapters.items():
        tok=AutoTokenizer.from_pretrained(adapter)
        model=AutoModelForCausalLM.from_pretrained(STUDENT,revision=BASE_REVISION,token=token,
            device_map={'':0},torch_dtype=torch.float16,quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
        model=PeftModel.from_pretrained(model,adapter);model.eval()
        path=ROOT/(name+'.json');rows=read(path,[])
        for q in questions[len(rows):]:
            chat=tok.apply_chat_template([dict(role='user',content=q['prompt'])],tokenize=False,add_generation_prompt=True)
            inputs=tok(chat,add_special_tokens=False,return_tensors='pt').to(model.device)
            with torch.inference_mode():
                out=model.generate(**inputs,max_new_tokens=160,do_sample=False,pad_token_id=tok.eos_token_id)[0][inputs['input_ids'].shape[-1]:]
            raw=tok.decode(out,skip_special_tokens=True).strip()
            rows.append(dict(q,raw=raw,generated_tokens=len(out),**grade(raw,q)))
            save(path,rows)
            if len(rows)%16==0:print('DIAG15 PROGRESS',name,len(rows),'/',len(questions),flush=True)
        all_results[name]=rows
        del model;gc.collect();torch.cuda.empty_cache()
    result=dict(protocol='Post-test paired diagnostic on 16 previously tested stories; 8 independent prompts per story; hints separately reported; no training or teacher calls.',adapter_hashes=HASHES,models=all_results)
    save(ROOT/'results.json',result)
    import gzip,base64
    print('DIAG15_EXPORT '+base64.b64encode(gzip.compress(json.dumps(result).encode())).decode(),flush=True)

if __name__=='__main__':main()
