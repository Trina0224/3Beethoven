"""Inference-only decomposition on exposed questions, not a new benchmark."""
import os, json, re, hashlib, shutil, importlib.metadata
from pathlib import Path
from fractions import Fraction as F
from stats_holdout_v0_6 import build
from stats_v0_3_common import prompt_for, parse_answer, digest
from run_stats_rotation_v1 import rotate
from flight_run_stats_v0_3 import STUDENT, read_json, save_json, package
from run_stats_v0_4 import BASE_REVISION

ROOT=Path('/kaggle/working/3beethoven_stats_diagnostic_v0_7')
ADAPTER_SHA='c0c259efb287bf279537d62a6cb63d5df1c325a5ec89d295cf5c1054ee5e0c61'

def aids(q):
    v=int(q['id'].split('_')[-1]); k=v+3; cat=q['category']
    if cat=='poisson':
        return ('For a Poisson process, the variance of a count equals rate times duration.',f'{k}*3') if v%2==0 else ('Var(aX+b)=a^2 Var(X); for Poisson X, Var(X)=E[X].',f'2^2*{k}')
    if cat=='expectation': return ('E[Y^2]=Var(Y)+(E[Y])^2; E[aX+b]=aE[X]+b; Var(aX+b)=a^2 Var(X).',f'4*2+(2*{k}+1)^2')
    if cat=='uniform': return ('A uniform variable conditioned to exceed an interior threshold is uniform from that threshold to its original upper endpoint; take the midpoint.',f'({2*k}+{4*k})/2')
    if cat=='type_i':
        p=F(v+1,20); return ('For three independent events of probability p, exactly two has probability 3*p^2*(1-p).',f'3*({p})^2*(1-({p}))')
    if cat=='type_ii':
        b=F(v+2,20); return ('If each of two independent methods misses with probability b, exactly one detection has probability 2*b*(1-b).',f'2*({b})*(1-({b}))')
    return ('The interval center stays fixed. Its half-width scales as 1/sqrt(n). The new upper endpoint is center plus new half-width.',f'({10*k}+{10*k+12})/2+6/3')

def tasks():
    out=[]
    for q in build():
        if int(q['id'].split('_')[-1]) not in (0,1,4,5): continue
        answer=q['choices']['ABCD'.index(q['answer_letter'])]; rule,expr=aids(q)
        base=dict(id=q['id'],category=q['category'],answer=answer)
        for mode in ('mc','mapping'):
            for shift in range(4):
                r=rotate(q,shift)
                prompt=prompt_for(r) if mode=='mc' else f'The correct numerical answer is {answer}. Which option equals that value?\n'+ '\n'.join(f'{a}. {b}' for a,b in zip('ABCD',r['choices']))+'\nReply with ONLY the letter A, B, C, or D.'
                out.append(dict(base,mode=mode,shift=shift,prompt=prompt,expected=r['answer_letter'],limit=16))
        for mode in ('free','steps','guided','arithmetic'):
            prompt=q['question']
            if mode=='guided': prompt+='\nUse this rule: '+rule
            if mode=='arithmetic': prompt='Evaluate this numerical expression: '+expr
            if mode=='steps': prompt+='\nShow a short calculation, then end with Answer: <number>.'
            else: prompt+='\nReply only with Answer: <number>. Use an exact fraction or decimal; do not explain.'
            out.append(dict(base,mode=mode,shift=0,prompt=prompt,expected=answer,limit=192 if mode=='steps' else 48))
    assert len(out)==288
    return out

def numeric(raw):
    text=raw.strip().replace('**','')
    matches=re.findall(r'(?:^|\n)\s*Answer:\s*([^\n]+)',text,re.I)
    candidate=matches[-1].strip() if matches else text
    if candidate.endswith('.'): candidate=candidate[:-1]
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*/\s*\d+)?',candidate): return None
    try: return F(candidate.replace(' ',''))
    except (ValueError,ZeroDivisionError): return None

def score(raw,t):
    if t['mode'] in ('mc','mapping'):
        pred=parse_answer(raw); return pred,pred==t['expected'],pred=='INVALID'
    pred=numeric(raw)
    correct=pred is not None and abs(float(pred-F(t['expected'])))<=max(1e-6,abs(float(F(t['expected'])))*1e-4)
    return str(pred) if pred is not None else 'INVALID',correct,pred is None

def main():
    import torch, kagglehub
    from kaggle_secrets import UserSecretsClient
    from peft import PeftModel,prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,set_seed
    ROOT.mkdir(exist_ok=True)
    ts=tasks(); protocol=dict(task_sha256=digest(ts),questions=24,tasks_per_model=288,models=['baseline','v05'],teacher_calls=0,training=False,base_revision=BASE_REVISION,adapter_sha256=ADAPTER_SHA,selection='Each topic indices 0,1,4,5; exposed v0.6 questions; fixed before diagnostic generation',numeric_tolerance='max(1e-6, abs(reference)*1e-4)',limits='Prompt interventions are diagnostic aids, not independent causal proof; no hidden-test claim')
    prior=read_json(ROOT/'protocol.json')
    if prior is not None and prior!=protocol: raise RuntimeError('Protocol mismatch')
    save_json(ROOT/'protocol.json',protocol);save_json(ROOT/'tasks.json',ts)
    saved=Path(kagglehub.notebook_output_download('trinashih/3beethoven-v0-2/versions/9'))
    adapter=saved/'3beethoven_stats_v0_5/adapter'
    assert hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest()==ADAPTER_SHA
    token=UserSecretsClient().get_secret('HF_TOKEN');os.environ['HF_TOKEN']=token
    tokenizer=AutoTokenizer.from_pretrained(adapter)
    set_seed(226)
    model=AutoModelForCausalLM.from_pretrained(STUDENT,revision=BASE_REVISION,token=token,device_map={'':0},torch_dtype=torch.float16,quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
    model=prepare_model_for_kbit_training(model,gradient_checkpointing_kwargs={'use_reentrant':False})
    summaries={}
    reference=read_json(Path(__file__).resolve().parent.parent/'docs/STATS_V0_6_RESULTS.json')
    for name in ('baseline','v05'):
        if name=='v05': model=PeftModel.from_pretrained(model,adapter)
        model.eval();model.config.use_cache=True
        path=ROOT/(name+'.json');rows=read_json(path,[])
        assert all(r['task_index']==i and r['prompt']==ts[i]['prompt'] for i,r in enumerate(rows))
        for i in range(len(rows),len(ts)):
            t=ts[i]
            chat=tokenizer.apply_chat_template([dict(role='user',content=t['prompt'])],tokenize=False,add_generation_prompt=True)
            inputs=tokenizer(chat,add_special_tokens=False,return_tensors='pt').to(model.device)
            with torch.inference_mode(): output=model.generate(**inputs,max_new_tokens=t['limit'],do_sample=False,pad_token_id=tokenizer.eos_token_id)
            tokens=output[0][inputs['input_ids'].shape[-1]:];raw=tokenizer.decode(tokens,skip_special_tokens=True).strip()
            pred,correct,invalid=score(raw,t)
            rows.append(dict(t,task_index=i,raw=raw,predicted=pred,correct=correct,invalid=invalid,generated_tokens=len(tokens),hit_token_limit=len(tokens)==t['limit']))
            save_json(path,rows)
            if len(rows)%24==0:print('DIAG PROGRESS',name,len(rows),'/',len(ts),flush=True)
        summaries[name]={mode:dict(n=len(g),correct=sum(r['correct'] for r in g),invalid=sum(r['invalid'] for r in g),hit_token_limit=sum(r['hit_token_limit'] for r in g)) for mode in ('mc','mapping','free','steps','guided','arithmetic') for g in [[r for r in rows if r['mode']==mode]]}
        old={(r['id'],r['shift']):r['raw'] for r in reference[name+'_new']}
        summaries[name]['mc_raw_reproduction']=dict(n=96,matches=sum(r['raw']==old[r['id'],r['shift']] for r in rows if r['mode']=='mc'))
        print('DIAG MODEL',name,json.dumps(summaries[name]),flush=True)
    save_json(ROOT/'summary.json',summaries)
    save_json(ROOT/'environment.json',dict(gpu=torch.cuda.get_device_name(0),packages={p:importlib.metadata.version(p) for p in ('torch','transformers','peft','bitsandbytes')}))
    shutil.copy2(__file__,ROOT/Path(__file__).name)
    package(ROOT)
    export=dict(protocol=protocol,summary=summaries,baseline=read_json(ROOT/'baseline.json'),v05=read_json(ROOT/'v05.json'),environment=read_json(ROOT/'environment.json'))
    archive=ROOT.with_suffix('.zip');export['archive']=dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    print('DIAG EXPORT',json.dumps(export),flush=True)

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['TOKENIZERS_PARALLELISM']='false';main()
