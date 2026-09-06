"""Verify saved diagnostic coverage, reference scoring, and archive integrity."""
import hashlib,json,zipfile
from collections import Counter
from diagnose_stats_v0_7 import ROOT,tasks,score

def main():
    ts=tasks(); summary=json.loads((ROOT/'summary.json').read_text())
    for model in ('baseline','v05'):
        rows=json.loads((ROOT/(model+'.json')).read_text())
        assert len(rows)==288
        assert Counter(r['mode'] for r in rows)==Counter(r['mode'] for r in ts)
        for i,(r,t) in enumerate(zip(rows,ts)):
            assert r['task_index']==i and all(r[k]==v for k,v in t.items())
            p,c,bad=score(r['raw'],t)
            assert (r['predicted'],r['correct'],r['invalid'])==(p,c,bad)
            assert r['hit_token_limit']==(r['generated_tokens']==t['limit'])
        for mode in ('mc','mapping','free','steps','guided','arithmetic'):
            group=[r for r in rows if r['mode']==mode]
            assert summary[model][mode]==dict(n=len(group),correct=sum(r['correct'] for r in group),invalid=sum(r['invalid'] for r in group),hit_token_limit=sum(r['hit_token_limit'] for r in group))
    archive=ROOT.with_suffix('.zip')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        manifest=json.loads(z.read('manifest.json'))
        for r in manifest:
            data=z.read(r['path'])
            assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256']
    print('DIAG VERIFIED',json.dumps(dict(responses=576,manifest_files=len(manifest),archive_bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())),flush=True)

if __name__=='__main__': main()
