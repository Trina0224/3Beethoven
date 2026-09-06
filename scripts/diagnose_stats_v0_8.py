"""Compact calculation prompts and completion of capped v0.7 answers."""
import os,json,hashlib,shutil
from pathlib import Path
from diagnose_stats_v0_7 import tasks as prior_tasks, score,ADAPTER_SHA
from flight_run_stats_v0_3 import STUDENT,read_json,save_json,package
from run_stats_v0_4 import BASE_REVISION
from stats_v0_3_common import digest
ROOT=Path('/kaggle/working/3beethoven_stats_diagnostic_v0_8')

def tasks():
    out=[]
    for old in prior_tasks():
        if old['mode'] not in ('free','arithmetic','steps'):continue
        r=dict(old)
        if old['mode']=='steps':r.update(mode='long_steps',limit=512)
        else:
            mode='compact' if old['mode']=='free' else 'compact_arithmetic'
            question=old['prompt'].split('\nReply only')[0]
            r.update(mode=mode,prompt=question+'\nGive a concise numerical solution in exactly three short lines.\nFormula: state the needed formula or operation.\nCalculation: substitute the numbers and calculate.\nAnswer: give only the final numerical value on this line.\nUse fractions when exact. No introduction or commentary.',limit=256)
        out.append(r)
    assert len(out)==72
    return out

def main():
    import torch,kagglehub
    from kaggle_secrets import UserSecretsClient
    from peft import PeftModel,prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,set_seed
    ROOT.mkdir(exist_ok=True);ts=tasks()
    previous=read_json(Path(__file__).resolve().parent.parent/'docs/STATS_DIAGNOSTIC_V0_7_RESULTS.json')
    protocol=dict(tasks_sha256=digest(ts),source_sha256=digest(previous),training=False,teacher_calls=0,models=['baseline','v05'],base_revision=BASE_REVISION,adapter_sha256=ADAPTER_SHA,reuse='Reuse completed v0.7 step outputs; rerun only capped step outputs with identical prompt and 512 budget',scope='24 exposed diagnostic questions, not a held-out evaluation')
    prior=read_json(ROOT/'protocol.json');assert prior is None or prior==protocol
    save_json(ROOT/'protocol.json',protocol);save_json(ROOT/'tasks.json',ts)
    saved=Path(kagglehub.notebook_output_download('trinashih/3beethoven-v0-2/versions/9'));adapter=saved/'3beethoven_stats_v0_5/adapter'
    assert hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest()==ADAPTER_SHA
    token=UserSecretsClient().get_secret('HF_TOKEN');os.environ['HF_TOKEN']=token
    tok=AutoTokenizer.from_pretrained(adapter);set_seed(226)
    model=AutoModelForCausalLM.from_pretrained(STUDENT,revision=BASE_REVISION,token=token,device_map={'':0},torch_dtype=torch.float16,quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
    model=prepare_model_for_kbit_training(model,gradient_checkpointing_kwargs={'use_reentrant':False})
    summary={}
    for name in ('baseline','v05'):
        if name=='v05':model=PeftModel.from_pretrained(model,adapter)
        model.eval();model.config.use_cache=True
        oldmap={(r['id'],r['mode']):r for r in previous[name]}
        path=ROOT/(name+'.json');rows=read_json(path,[])
        for i in range(len(rows),len(ts)):
            t=ts[i];old=oldmap[t['id'],'steps'] if t['mode']=='long_steps' else None
            reused=old is not None and not old['hit_token_limit']
            if reused:raw=old['raw'];n=old['generated_tokens']
            else:
                chat=tok.apply_chat_template([dict(role='user',content=t['prompt'])],tokenize=False,add_generation_prompt=True)
                inp=tok(chat,add_special_tokens=False,return_tensors='pt').to(model.device)
                with torch.inference_mode():generated=model.generate(**inp,max_new_tokens=t['limit'],do_sample=False,pad_token_id=tok.eos_token_id)[0][inp['input_ids'].shape[-1]:]
                raw=tok.decode(generated,skip_special_tokens=True).strip();n=len(generated)
            pred,correct,invalid=score(raw,t)
            rows.append(dict(t,task_index=i,raw=raw,predicted=pred,correct=correct,invalid=invalid,generated_tokens=n,hit_token_limit=n==t['limit'],reused=reused,prior_prefix_match=None if old is None else raw.startswith(old['raw'])))
            save_json(path,rows)
            if len(rows)%12==0: print('V08 PROGRESS',name,len(rows),'/72',flush=True)
        summary[name]={mode:dict(n=len(g),correct=sum(r['correct'] for r in g),invalid=sum(r['invalid'] for r in g),hit_token_limit=sum(r['hit_token_limit'] for r in g),reused=sum(r['reused'] for r in g)) for mode in ('compact','compact_arithmetic','long_steps') for g in [[r for r in rows if r['mode']==mode]]}
        print('V08 MODEL',name,json.dumps(summary[name]),flush=True)
    save_json(ROOT/'summary.json',summary);shutil.copy2(__file__,ROOT/Path(__file__).name);package(ROOT)
    export=dict(protocol=protocol,summary=summary,baseline=read_json(ROOT/'baseline.json'),v05=read_json(ROOT/'v05.json'))
    archive=ROOT.with_suffix('.zip');export['archive']=dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    print('V08 EXPORT',json.dumps(export),flush=True)

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['TOKENIZERS_PARALLELISM']='false';main()
