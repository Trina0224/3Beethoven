"""Verify frozen references, stage transitions, matched exposures and saved weights."""
import json,hashlib,zipfile,importlib.metadata
from collections import Counter
from pathlib import Path
from run_stats_v0_18 import ROOT,DATA_SHA,metrics,stage_key,mastery
from stats_curriculum_v0_18 import build,digest,prompt,score,stage_rows
from flight_run_stats_v0_3 import save_json as save,read_json as read,package

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    import torch
    from safetensors.torch import load_file
    data=build();assert digest(data)==DATA_SHA and data==read(ROOT/'frozen_questions.json')
    summary=read(ROOT/'summary.json');stages=summary['stages'];outputs={};histories={};pending=[]
    for p in sorted(ROOT.rglob('*validation.json'))+sorted(ROOT.glob('*_test.json')):
        split='test' if p.name.endswith('_test.json') else 'validation';qs={q['id']:q for q in data[split]}
        rows=read(p);assert len(rows)==len(qs) and {r['id'] for r in rows}==set(qs)
        for r in rows:
            q=qs[r['id']];assert r['question_sha256']==digest(q) and r['prompt']==prompt(q)
            assert all(r[k]==v for k,v in score(r['raw'],q).items())
            if r['review_required']:pending.append(dict(file=str(p.relative_to(ROOT)),id=r['id'],question=q['question'],reference=q['expression'],raw=r['raw'],computed=r.get('computed')))
        outputs[str(p.relative_to(ROOT))]=rows
    assert metrics(outputs['baseline_validation.json'])==summary['baseline']
    for name,m in summary['tests'].items():assert metrics(outputs[name+'_test.json'])==m
    exposure=[]
    for s in stages:
        stage=s['stage'];folder=ROOT/f'stage_{stage}';history=read(folder/'history.json');histories[str(stage)]=history
        assert [h['epoch'] for h in history]==list(range(1,s['epochs']+1))
        for h in history:
            assert metrics(outputs[f'stage_{stage}/epoch_{h["epoch"]}_validation.json'])==h['metrics']
            assert mastery(h['metrics'],stage)==h['mastery']
        last=history[-1];assert last['metrics']==s['validation']
        if s['stop_reason']=='stable_mastery':assert len(history)>=3 and all(h['mastery'] for h in history[-2:])
        elif s['stop_reason']=='validation_plateau':
            best=max(range(len(history)),key=lambda i:stage_key(history[i]['metrics'],stage));assert len(history)-1-best>=3
        else:assert s['stop_reason']=='budget_cap_without_convergence' and s['epochs']==8 and stage==len(stages)
        rows=stage_rows(data,stage);assert rows==read(folder/'train.json');exposure+=rows*s['epochs']
        assert s['steps']==len(rows)*s['epochs']//8
        assert sha(folder/'adapter/adapter_model.safetensors')==s['adapter_sha256']
    assert exposure==read(ROOT/'curriculum_exposure.json')
    control=read(ROOT/'control_exposure.json')
    assert Counter(digest(r) for r in control)==Counter(digest(r) for r in exposure)
    for s,c in zip(stages,summary['controls']):
        assert s['steps']==c['steps']
        assert metrics(outputs[f'control_{c["stage"]}/validation.json'])==c['validation']
        assert sha(ROOT/f'control_{c["stage"]}'/'adapter/adapter_model.safetensors')==c['adapter_sha256']
    weights={}
    for p in sorted(ROOT.glob('*/adapter/adapter_model.safetensors')):
        tensors=load_file(str(p));assert len(tensors)==392 and all(torch.isfinite(t).all() for t in tensors.values())
        weights[str(p.relative_to(ROOT))]=dict(sha256=sha(p),finite_tensors=len(tensors))
    # Full checkpoints remain in Kaggle output; ZIP includes separate stage adapters.
    checkpoints=[]
    for p in sorted(ROOT.glob('stage_*/checkpoints/checkpoint-*')):
        required=['adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','trainer_state.json']
        assert all((p/n).is_file() for n in required) and any(p.glob('rng_state*.pth'))
        checkpoints.append(str(p.relative_to(ROOT)))
    assert len(checkpoints)==sum(s['epochs'] for s in stages)
    environment=dict(gpu=torch.cuda.get_device_name(0),packages={p:importlib.metadata.version(p) for p in ('torch','transformers','peft','bitsandbytes','datasets','accelerate')})
    save(ROOT/'environment.json',environment)
    result=dict(summary=summary,histories=histories,outputs=outputs,pending_semantic_review=pending,environment=environment,
                verification=dict(responses=sum(len(v) for v in outputs.values()),matched_training_exposures=len(exposure),weights=weights,stage_checkpoints=checkpoints,
                                  truncated=sum(r['hit_token_limit'] for rows in outputs.values() for r in rows)))
    save(ROOT/'verified_results.json',result)
    if (ROOT/'review_credits.json').exists():
        from review_stats_v0_18 import main as review_main
        review_main()
    package(ROOT)
    manifest=read(ROOT/'manifest.json')
    for r in manifest:
        p=ROOT/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
    archive=ROOT.with_suffix('.zip')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p,m in weights.items():assert hashlib.sha256(z.read(p)).hexdigest()==m['sha256']
    receipt=dict(archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),manifest_files=len(manifest),crc_verified=True,
                 full_epoch_checkpoints='Saved in Kaggle directory output, excluded from compact ZIP; all boundary adapters included in ZIP')
    save(ROOT.parent/'v18_backup_receipt.json',receipt)
    print('V18 VERIFIED',dict(**result['verification'],pending_review=len(pending),archive=receipt),flush=True)

if __name__=='__main__':
    from transfer_stats_v0_18 import main as transfer_main
    transfer_main()
    main()
