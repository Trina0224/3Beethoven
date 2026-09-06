"""Exact delta interpolation and conservative balanced response rehearsal."""
import os,gc,json,hashlib,shutil
from pathlib import Path
from flight_run_stats_v0_3 import save_json as save,read_json as read,package,STUDENT
from stats_curriculum_v0_17 import build,digest,prompt,score,selection_key
ROOT=Path('/kaggle/working/3beethoven_stats_v0_17')
P15=ROOT.parent/'3beethoven_stats_v0_15';P16=ROOT.parent/'3beethoven_stats_v0_16'
HASHES={15:'9369d52de4a886df9da0c872cd41bd4e01af0a38bf02ad724b5951c1a6b9f5d3',16:'117a009f72ebafe6e6baefef62a6b81e7bbcefbc902f7eb3d93f5e73f48d46d0'}
DATA_SHA='6b686160203e15ca9cc6be270ec2fe98efeccfcf961be560505af3d2eb477908'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def restore():
    cache=None
    for v,root in ((15,P15),(16,P16)):
        target=root/'adapter/adapter_model.safetensors'
        if not target.exists():
            if cache is None:
                import kagglehub
                cache=Path(kagglehub.notebook_output_download('trinashih/3beethoven-v0-2/versions/32'))
            candidates=[p for p in cache.rglob(root.name) if (p/'adapter/adapter_model.safetensors').exists()]
            assert candidates,f'Missing saved v{v}'
            src=min(candidates,key=lambda p:len(p.parts))
            shutil.copytree(src/'adapter',root/'adapter')
        assert sha(target)==HASHES[v],f'Wrong v{v} weights'
    repo=Path(__file__).resolve().parents[1]
    save(P15/'teacher/records.json',read(repo/'docs/STATS_V0_15_TEACHER_INITIAL.json'))
    save(P15/'teacher/focused_supplements.json',read(repo/'docs/STATS_V0_15_RESULTS.json')['focused_supplements'])
    save(P16/'teacher/records.json',read(repo/'docs/STATS_V0_16_RESULTS.json')['teacher_records'])
    print('V17 RESTORE VERIFIED',flush=True)

def mixed_state(s15,s16,alpha):
    """Concatenate factors: Bmix Amix = (1-a) B15 A15 + a B16 A16.
    Never average A and B separately (which introduces cross terms).
    """
    import torch
    assert s15.keys()==s16.keys();out={}
    for k in s15:
        assert s15[k].shape==s16[k].shape
        if '.lora_A.' in k:out[k]=torch.cat([s15[k],s16[k]],dim=0).contiguous()
        elif '.lora_B.' in k:out[k]=torch.cat([(1-alpha)*s15[k],alpha*s16[k]],dim=1).contiguous()
        else:raise ValueError('Unexpected adapter parameter '+k)
    return out

def mix_adapters():
    from safetensors.torch import load_file,save_file
    c15=read(P15/'adapter/adapter_config.json');c16=read(P16/'adapter/adapter_config.json')
    for key in ('r','lora_alpha','target_modules','base_model_name_or_path','rank_pattern','alpha_pattern','use_rslora','use_dora','bias'):
        assert c15.get(key)==c16.get(key),(key,c15.get(key),c16.get(key))
    assert not c15.get('rank_pattern') and not c15.get('alpha_pattern') and not c15.get('use_rslora') and not c15.get('use_dora')
    assert c15.get('bias')=='none'
    a,b=load_file(str(P15/'adapter/adapter_model.safetensors')),load_file(str(P16/'adapter/adapter_model.safetensors'))
    out={}
    for weight in (.25,.5,.75):
        name='mix_'+str(int(weight*100));folder=ROOT/name/'adapter';folder.mkdir(parents=True,exist_ok=True)
        state=mixed_state(a,b,weight)
        # Numerical identity checked on every layer using a small deterministic probe.
        import torch
        gen=torch.Generator().manual_seed(1717)
        for key in a:
            if '.lora_A.' not in key:continue
            bk=key.replace('.lora_A.','.lora_B.');x=torch.randn(a[key].shape[1],2,generator=gen)
            expected=(1-weight)*(a[bk].float()@(a[key].float()@x))+weight*(b[bk].float()@(b[key].float()@x))
            actual=state[bk].float()@(state[key].float()@x)
            assert torch.allclose(actual,expected,atol=1e-5,rtol=1e-4),key
        save_file(state,str(folder/'adapter_model.safetensors'))
        config=dict(c15,r=c15['r']*2,lora_alpha=c15['lora_alpha']*2,inference_mode=True)
        save(folder/'adapter_config.json',config)
        for p in (P15/'adapter').iterdir():
            if p.name not in ('adapter_config.json','adapter_model.safetensors') and p.is_file():shutil.copy2(p,folder/p.name)
        save(folder.parent/'merge.json',dict(alpha_v16=weight,method='exact concatenated low-rank delta sum',source_hashes=HASHES,rank=config['r'],adapter_sha256=sha(folder/'adapter_model.safetensors')))
        out[name]=folder
    return out

def training_rows(repo):
    from prepare_stats_v0_15 import teacher_rows
    from run_stats_v0_16 import examples
    from stats_curriculum_v0_16 import build as build16
    from stats_curriculum_v0_13 import KINDS
    v15=teacher_rows(repo)['train'];old=read(repo/'docs/STATS_V0_14_VERIFIED_DISTILLATION.json')['train']
    selected=[]
    for kind in KINDS:
        pool={r['source_id']:r for r in v15+old if f'_{kind}_' in r['source_id'] and not r['source_id'].endswith(('_mean','_variance'))}
        rows=list(pool.values());assert len(rows)>=24,(kind,len(rows))
        selected.extend(dict(r,category=kind) for r in rows[:24])
    v16,_=examples(build16(),repo)
    # Two families x four contrast tasks x eight stories, all prior TRAIN rows.
    for family in ('moment','poisson_scaled'):
        for task in ('mean','variance','moment','scale'):
            rows=[r for r in v16['train'] if r['q']['family']==family and r['q']['task']==task][:8]
            assert len(rows)==8
            selected.extend(dict(source_id=r['q']['id'],prompt=prompt(r['q']),target=r['target'],teacher_raw=r['teacher_raw'],category=family+'_'+task) for r in rows)
    assert len(selected)==256 and len({r['source_id'] for r in selected})==256
    assert all(r.get('teacher_raw') for r in selected)
    save(ROOT/'train.json',selected);return selected

def evaluate(model,tok,qs,path):
    import torch
    rows=read(path,[]);lookup={q['id']:q for q in qs}
    assert len({r['id'] for r in rows})==len(rows)
    for r in rows:assert r['question_sha256']==digest(lookup[r['id']]) and r['prompt']==prompt(lookup[r['id']])
    done={r['id'] for r in rows};model.eval();model.config.use_cache=True
    for q in qs:
        if q['id'] in done:continue
        p=prompt(q);chat=tok.apply_chat_template([dict(role='user',content=p)],tokenize=False,add_generation_prompt=True)
        inputs=tok(chat,add_special_tokens=False,return_tensors='pt').to(model.device)
        with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=160,do_sample=False,pad_token_id=tok.eos_token_id)[0][inputs['input_ids'].shape[-1]:]
        raw=tok.decode(out,skip_special_tokens=True).strip()
        rows.append(dict(id=q['id'],category=q['category'],question_sha256=digest(q),prompt=p,raw=raw,generated_tokens=len(out),hit_token_limit=len(out)==160,**score(raw,q)))
        save(path,rows)
        if len(rows)%12==0:print('V17 EVAL',path.stem,len(rows),'/',len(qs),flush=True)
    return dict(n=len(rows),correct=sum(r['correct'] for r in rows),pending=sum(r['review_required'] for r in rows),by_category={c:dict(n=sum(r['category']==c for r in rows),correct=sum(r['correct'] for r in rows if r['category']==c)) for c in sorted({q['category'] for q in qs})})

def main():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,Trainer,TrainingArguments,set_seed
    from peft import PeftModel,prepare_model_for_kbit_training
    from kaggle_secrets import UserSecretsClient
    from run_stats_v0_4 import BASE_REVISION,dataset,evaluate as old_eval
    from flight_run_stats_v0_1 import CausalCollator
    from stats_holdout_v1 import questions as old_questions
    ROOT.mkdir(exist_ok=True);repo=Path(__file__).resolve().parents[1];data=build()
    assert digest(data)==DATA_SHA and data==read(repo/'docs/STATS_V0_17_FROZEN_QUESTIONS.json')
    protocol=dict(data_sha256=DATA_SHA,source_hashes=HASHES,seed=1717,lr=5e-6,max_steps=32,checkpoints=[8,16,32],new_teacher_calls=0,
        prompt='Explicit numerical-expression grammar including comb(n,r); identical for every candidate',
        selection='48 fresh balanced validation questions; total correct, then non-target correct, then minimum family correct; remaining ties prefer earlier candidate',
        candidates=['v15','mix_25','mix_50','mix_75','v16','train_8','train_16','train_32'],test_count=96,
        merge='Exact low-rank delta concatenation, double rank; no cross terms; no claim of unchanged memory',
        promotion='New-test improvement >= 6/96 over v15; at most 3 losses in six non-target families; old MC within 4/240. Report all endpoints separately.')
    prior=read(ROOT/'protocol.json');assert prior is None or prior==protocol
    save(ROOT/'protocol.json',protocol);save(ROOT/'frozen_questions.json',data)
    source=ROOT/'source';source.mkdir(exist_ok=True)
    for p in (repo/'scripts').glob('*.py'):shutil.copy2(p,source/p.name)
    restore();mixes=mix_adapters();train=training_rows(repo)
    token=UserSecretsClient().get_secret('HF_TOKEN');tok=AutoTokenizer.from_pretrained(P15/'adapter')
    def base(training=False):
        set_seed(1717)
        m=AutoModelForCausalLM.from_pretrained(STUDENT,revision=BASE_REVISION,token=token,device_map={'':0},torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
        return prepare_model_for_kbit_training(m,gradient_checkpointing_kwargs={'use_reentrant':False}) if training else m
    if not read(ROOT/'training_complete.json'):
        model=PeftModel.from_pretrained(base(True),P15/'adapter',is_trainable=True);model.config.use_cache=False
        args=TrainingArguments(output_dir=str(ROOT/'checkpoints'),max_steps=32,per_device_train_batch_size=1,gradient_accumulation_steps=8,
            learning_rate=5e-6,warmup_steps=2,lr_scheduler_type='cosine',fp16=True,logging_steps=8,save_strategy='steps',save_steps=8,
            save_total_limit=4,report_to='none',remove_unused_columns=False,optim='paged_adamw_8bit',seed=1717,disable_tqdm=True)
        trainer=Trainer(model=model,args=args,train_dataset=dataset(train,tok),data_collator=CausalCollator(tok))
        ckpts=sorted((ROOT/'checkpoints').glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
        result=trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)
        for n in (8,16,32):
            dst=ROOT/f'train_{n}'/'adapter';dst.mkdir(parents=True,exist_ok=True)
            for name in ('adapter_config.json','adapter_model.safetensors'):shutil.copy2(ROOT/'checkpoints'/f'checkpoint-{n}'/name,dst/name)
            tok.save_pretrained(dst)
        save(ROOT/'training_complete.json',dict(steps=trainer.state.global_step,loss=result.training_loss,log=trainer.state.log_history))
        del trainer,model;gc.collect();torch.cuda.empty_cache()
        print('V17 TRAINING COMPLETE 32 STEPS',flush=True)
    paths={'v15':P15/'adapter',**mixes,'v16':P16/'adapter',**{f'train_{n}':ROOT/f'train_{n}'/'adapter' for n in (8,16,32)}}
    vals={}
    for name in protocol['candidates']:
        path=ROOT/(name+'_validation.json')
        if len(read(path,[]))==48:
            rows=read(path);vals[name]=dict(n=48,correct=sum(r['correct'] for r in rows),by_category={c:dict(n=6,correct=sum(r['correct'] for r in rows if r['category']==c)) for c in {q['category'] for q in data['validation']}})
        else:
            model=PeftModel.from_pretrained(base(),paths[name]);vals[name]=evaluate(model,tok,data['validation'],path)
            del model;gc.collect();torch.cuda.empty_cache()
        print('V17 VALIDATION',name,vals[name],flush=True);save(ROOT/'validation_metrics.json',vals)
    best_mix=max(['mix_25','mix_50','mix_75'],key=lambda n:selection_key(vals[n]))
    best_train=max(['train_8','train_16','train_32'],key=lambda n:selection_key(vals[n]))
    winner=max(protocol['candidates'],key=lambda n:selection_key(vals[n]))
    selection=dict(best_mix=best_mix,best_train=best_train,winner=winner,validation=vals)
    prior=read(ROOT/'selection.json');assert prior is None or prior==selection
    save(ROOT/'selection.json',selection);shutil.copytree(paths[winner],ROOT/'adapter',dirs_exist_ok=True)
    package(ROOT);print('V17 SELECTION FROZEN',winner,best_mix,best_train,flush=True)
    tests={}
    for name in dict.fromkeys(['v15','v16',best_mix,best_train]):
        model=PeftModel.from_pretrained(base(),paths[name]);tests[name]=evaluate(model,tok,data['test'],ROOT/(name+'_test.json'))
        if name in ('v15',winner):tests[name]['old']=old_eval(model,tok,old_questions(),ROOT/(name+'_old.json'))
        save(ROOT/'test_metrics.json',tests);print('V17 TEST RESULT',name,tests[name],flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    save(ROOT/'summary.json',dict(protocol=protocol,selection=selection,tests=tests,adapter_sha256=sha(ROOT/'adapter/adapter_model.safetensors'),training=read(ROOT/'training_complete.json')))
    package(ROOT);print('V17 COMPLETE',flush=True)

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['TOKENIZERS_PARALLELISM']='false';main()
