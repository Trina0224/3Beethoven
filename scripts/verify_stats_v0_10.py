"""Verify raw evaluation, selected tensors, data gates and complete ZIP."""
import json,hashlib,zipfile
from fractions import Fraction as F
from flight_run_stats_v0_3 import read_json
from stats_curriculum_v0_10 import build,numeric_score
from stats_holdout_v1 import questions as old_questions
from stats_v0_3_common import parse_answer
from diagnose_stats_v0_7 import score
from generate_stats_v0_10 import ROOT

def main():
    data=build();summary=read_json(ROOT/'summary.json')
    export=dict(summary=summary,environment=read_json(ROOT/'environment.json'),trainer_log=read_json(ROOT/'trainer_log.json'),teacher=read_json(ROOT/'teacher_test.json'),audit=read_json(ROOT/'audit_approved.json'))
    total=0
    for name in ('baseline','v09','v10'):
        for suite in ('mc','numeric')+(('old',) if name in ('v09','v10') else ()):
            qs=data['test'] if suite!='old' else old_questions();qmap={q['id']:q for q in qs};rows=read_json(ROOT/(name+'_'+suite+'.json'))
            if suite=='numeric':
                assert len(rows)==48 and {r['id'] for r in rows}==set(qmap)
                for r in rows:
                    assert r['expected']==qmap[r['id']]['answer']
                    pred,c,bad=numeric_score(r['raw'],r['expected'])
                    assert (pred,c,bad)==(r['predicted'],r['correct'],r['invalid'])
                assert sum(r['correct'] for r in rows)==summary['models'][name][suite]['correct']
            else:
                assert len(rows)==4*len(qs)
                assert {(r['id'],r['shift']) for r in rows}=={(q['id'],s) for q in qs for s in range(4)}
                for r in rows:
                    q=qmap[r['id']];s=r['shift'];cs=q['choices'][s:]+q['choices'][:s]
                    answer=q['choices']['ABCD'.index(q['answer_letter'])];expected='ABCD'[cs.index(answer)]
                    assert expected==r['expected'] and parse_answer(r['raw'])==r['predicted'] and r['correct']==(r['predicted']==expected)
                assert sum(r['correct'] for r in rows)==summary['models'][name][suite]['overall']['correct']
            total+=len(rows);export[name+'_'+suite]=rows
    assert total==1200
    import torch
    from safetensors import safe_open
    with safe_open(ROOT/'adapter/adapter_model.safetensors',framework='pt',device='cpu') as f:
        keys=list(f.keys());assert keys and all(torch.isfinite(f.get_tensor(k)).all().item() for k in keys)
    archive=ROOT.with_suffix('.zip')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;manifest=json.loads(z.read('manifest.json'))
        for row in manifest:
            b=z.read(row['path']);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
    export['verification']=dict(responses=total,finite_tensors=len(keys),manifest_files=len(manifest),archive_bytes=archive.stat().st_size,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    print('V10 VERIFIED',json.dumps(export['verification']),flush=True)
    print('V10 EXPORT',json.dumps(export),flush=True)

if __name__=='__main__':main()
