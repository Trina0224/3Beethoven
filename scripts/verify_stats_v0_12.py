"""Verify v0.12 outputs and emit all raw responses for independent review."""
import hashlib,json,zipfile
from pathlib import Path
from flight_run_stats_v0_3 import read_json
from run_stats_v0_4 import metrics
from run_stats_v0_12 import ROOT
from stats_curriculum_v0_12 import build,prompt,score
from stats_holdout_v1 import questions as old_questions

def main():
    import torch
    from safetensors.torch import load_file
    data=build();summary=read_json(ROOT/'summary.json');assert data==read_json(ROOT/'frozen_questions.json');output=dict(summary=summary,environment=read_json(ROOT/'environment.json'),trainer_log=read_json(ROOT/'trainer_log.json'));count=0
    for model in ('v10','v11','v12'):
        for label,key in (('micro','micro_test'),('transfer','transfer_test')):
            rows=read_json(ROOT/f'{model}_{label}.json');qs={q['id']:q for q in data[key]};assert len(rows)==len(qs) and {r['id'] for r in rows}==set(qs)
            for r in rows:
                q=qs[r['id']];assert r['expected']==q['answer'] and r['prompt']==prompt(q);assert all(r[k]==v for k,v in score(r['raw'],q['answer'],q['category']).items())
            expected=summary['models'][model][label]['overall'];assert expected['correct']==sum(r['correct'] for r in rows);output[f'{model}_{label}']=rows;count+=len(rows)
        old=read_json(ROOT/f'{model}_old.json');assert metrics(old,old_questions())==summary['models'][model]['old'];output[f'{model}_old']=old;count+=len(old)
    assert count==1104
    adapter=ROOT/'adapter/adapter_model.safetensors';weights=load_file(str(adapter));assert len(weights)==392 and all(torch.isfinite(v).all() for v in weights.values());assert hashlib.sha256(adapter.read_bytes()).hexdigest()==summary['adapter_sha256']
    manifest=read_json(ROOT/'manifest.json')
    for r in manifest:
        p=ROOT/r['path'];assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
    archive=ROOT.with_suffix('.zip')
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None and hashlib.sha256(z.read('adapter/adapter_model.safetensors')).hexdigest()==summary['adapter_sha256']
    output['verification']=dict(responses=count,finite_tensors=len(weights),manifest_files=len(manifest),archive_bytes=archive.stat().st_size,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    print('V12 VERIFIED',json.dumps(output['verification']),flush=True);print('V12 EXPORT',json.dumps(output),flush=True)

if __name__=='__main__':main()
