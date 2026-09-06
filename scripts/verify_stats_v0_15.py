"""Verify every saved prediction and adapter before reporting a completed run."""
import hashlib
import json
import zipfile
from flight_run_stats_v0_3 import read_json
from run_stats_v0_4 import metrics
from run_stats_v0_15 import ROOT,build,prompt,score
from stats_holdout_v1 import questions as old_questions


def main():
    import torch
    from safetensors.torch import load_file
    data=build();summary=read_json(ROOT/'summary.json')
    assert data==read_json(ROOT/'frozen_questions.json')
    output=dict(summary=summary,environment=read_json(ROOT/'environment.json'),trainer_log=read_json(ROOT/'trainer_log.json'))
    count=0
    for model in ('baseline','v14','v15'):
        rows=read_json(ROOT/f'{model}_formulation.json');qs={q['id']:q for q in data['test']}
        assert len(rows)==len(qs) and {r['id'] for r in rows}==set(qs)
        for r in rows:
            q=qs[r['id']]
            assert r['prompt']==prompt(q)
            assert all(r[k]==v for k,v in score(r['raw'],q).items())
        assert summary['models'][model]['formulation']['overall']['correct']==sum(r['correct'] for r in rows)
        output[f'{model}_formulation']=rows;count+=len(rows)
        old=read_json(ROOT/f'{model}_old.json')
        assert metrics(old,old_questions())==summary['models'][model]['retention']
        output[f'{model}_old']=old;count+=len(old)
    assert count==912
    adapter=ROOT/'adapter/adapter_model.safetensors';weights=load_file(str(adapter))
    assert len(weights)==392 and all(torch.isfinite(v).all() for v in weights.values())
    assert hashlib.sha256(adapter.read_bytes()).hexdigest()==summary['adapter_sha256']
    manifest=read_json(ROOT/'manifest.json')
    for r in manifest:
        p=ROOT/r['path']
        assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
    archive=ROOT.with_suffix('.zip')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert hashlib.sha256(z.read('adapter/adapter_model.safetensors')).hexdigest()==summary['adapter_sha256']
    output['verification']=dict(responses=count,finite_tensors=len(weights),manifest_files=len(manifest),archive_bytes=archive.stat().st_size,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    print('V15 VERIFIED',json.dumps(output['verification']),flush=True)
    # Keep large prediction exports out of the historical notebook DOM.
    (ROOT/'verified_results.json').write_text(json.dumps(output,indent=2)+'\n')
    print('V15 RESULTS SAVED',str(ROOT/'verified_results.json'),flush=True)


if __name__=='__main__':main()
