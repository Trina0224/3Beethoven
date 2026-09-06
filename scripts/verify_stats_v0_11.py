"""Verify saved outputs and export raw responses for an independent review."""
import json,hashlib,zipfile
from pathlib import Path
from fractions import Fraction as F
from flight_run_stats_v0_3 import read_json
from stats_curriculum_v0_11 import build,prompt,score
from run_stats_v0_11 import ROOT
from run_stats_v0_4 import metrics
from stats_holdout_v1 import questions as old_questions

def main():
    import torch
    from safetensors.torch import load_file
    data=build();summary=read_json(ROOT/'summary.json');assert data==read_json(ROOT/'frozen_questions.json')
    output=dict(summary=summary,environment=read_json(ROOT/'environment.json'),trainer_log=read_json(ROOT/'trainer_log.json'))
    paired={};count=0
    for model in ('v10','v11'):
        for label,key in (('arithmetic','test'),('variants','test_variants'),('transfer','transfer')):
            rows=read_json(ROOT/f'{model}_{label}.json');qs={q['id']:q for q in data[key]}
            assert len(rows)==len(qs) and {r['id'] for r in rows}==set(qs)
            for r in rows:
                q=qs[r['id']];assert r['expected']==q['answer'] and r['prompt']==prompt(q)
                assert all(r[k]==v for k,v in score(r['raw'],q['answer'],q['category']).items())
            expected=summary['models'][model][label]['overall']
            assert expected['strict']==sum(r['correct'] for r in rows) and expected['reviewed']==sum(r['reviewed_correct'] for r in rows)
            output[f'{model}_{label}']=rows;count+=len(rows)
        old=read_json(ROOT/f'{model}_old.json');assert metrics(old,old_questions())==summary['models'][model]['old']
        output[f'{model}_old']=old;count+=len(old)
        a=output[f'{model}_arithmetic'];b=output[f'{model}_variants'];lookup={r['id']:r for r in b}
        paired[model]=dict(n=60,both_correct=sum(r['reviewed_correct'] and lookup[r['id']+'_variant']['reviewed_correct'] for r in a),canonical_only=sum(r['reviewed_correct'] and not lookup[r['id']+'_variant']['reviewed_correct'] for r in a),variant_only=sum(not r['reviewed_correct'] and lookup[r['id']+'_variant']['reviewed_correct'] for r in a))
    assert count==816
    adapter=ROOT/'adapter/adapter_model.safetensors';weights=load_file(str(adapter))
    assert len(weights)==392 and all(torch.isfinite(v).all() for v in weights.values())
    assert hashlib.sha256(adapter.read_bytes()).hexdigest()==summary['adapter_sha256']
    manifest=read_json(ROOT/'manifest.json')
    for r in manifest:
        p=ROOT/r['path'];assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
    archive=ROOT.with_suffix('.zip')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert hashlib.sha256(z.read('adapter/adapter_model.safetensors')).hexdigest()==summary['adapter_sha256']
    output['paired']=paired
    output['verification']=dict(responses=count,finite_tensors=len(weights),manifest_files=len(manifest),archive_bytes=archive.stat().st_size,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    print('V11 VERIFIED',json.dumps(output['verification']),flush=True)
    print('V11 EXPORT',json.dumps(output),flush=True)

if __name__=='__main__':main()
