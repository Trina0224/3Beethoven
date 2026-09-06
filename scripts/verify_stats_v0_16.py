"""Verify all responses, weights and archived bytes; export compact results."""
import base64,gzip,hashlib,json,zipfile
from run_stats_v0_16 import ROOT,build,prompt,score,read,save,digest

def main():
    import torch
    from safetensors.torch import load_file
    from run_stats_v0_4 import metrics
    from stats_holdout_v1 import questions as old_questions
    data=build();summary=read(ROOT/'summary.json');out=dict(summary=summary,environment=read(ROOT/'environment.json'),
        teacher_records=read(ROOT/'teacher/records.json'),teacher_structural_review=read(ROOT/'teacher/structural_review.json'),training_logs={s:read(ROOT/s/'log.json') for s in ('full','cue','none')})
    count=0
    for name in ('v15','v16'):
        for condition in ('unaided','aided'):
            qs={q['id']:q for q in data['test'] if condition=='unaided' or q['category'] in ('moment','poisson_scaled')}
            rows=read(ROOT/(name+'_'+condition+'.json'))
            assert len(rows)==len(qs) and {r['id'] for r in rows}==set(qs)
            for r in rows:
                q=qs[r['id']];assert r['prompt']==prompt(q,'none' if condition=='unaided' else 'full') and r['question_sha256']==digest(q)
                assert all(r[k]==v for k,v in score(r['raw'],q).items())
            assert sum(r['correct'] for r in rows)==summary['models'][name][condition]['correct']
            out[name+'_'+condition]=rows;count+=len(rows)
        rows=read(ROOT/(name+'_old.json'));assert metrics(rows,old_questions())==summary['models'][name]['retention']
        out[name+'_old']=rows;count+=len(rows)
    assert count==720
    weights=load_file(str(ROOT/'adapter/adapter_model.safetensors'))
    assert len(weights)==392 and all(torch.isfinite(x).all() for x in weights.values())
    manifest=read(ROOT/'manifest.json')
    for r in manifest:
        p=ROOT/r['path'];assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
    archive=ROOT.with_suffix('.zip')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert hashlib.sha256(z.read('adapter/adapter_model.safetensors')).hexdigest()==summary['adapter_sha256']
    out['verification']=dict(responses=count,finite_tensors=392,manifest_files=len(manifest),archive_bytes=archive.stat().st_size,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    save(ROOT/'verified_results.json',out)
    print('V16 VERIFIED',out['verification'],flush=True)
    print('V16_EXPORT '+base64.b64encode(gzip.compress(json.dumps(out).encode())).decode(),flush=True)

if __name__=='__main__':main()
