"""Verify frozen selection, response provenance, finite weights and archive."""
import json,hashlib,zipfile
from run_stats_v0_17 import ROOT,read,save,sha,DATA_SHA
from stats_curriculum_v0_17 import build,digest,prompt,score,selection_key

def main():
 import torch
 from safetensors.torch import load_file
 from run_stats_v0_4 import metrics
 from stats_holdout_v1 import questions as old_questions
 data=build();assert digest(data)==DATA_SHA
 summary=read(ROOT/'summary.json');selection=summary['selection'];protocol=summary['protocol']
 count=0;outputs={};validation={}
 for split,names in [('validation',protocol['candidates']),('test',list(summary['tests']))]:
  qs={q['id']:q for q in data[split]}
  for name in names:
   rows=read(ROOT/(name+'_'+split+'.json'));assert len(rows)==len(qs) and {r['id'] for r in rows}==set(qs)
   for r in rows:
    q=qs[r['id']];assert r['question_sha256']==digest(q) and r['prompt']==prompt(q)
    assert all(r[k]==v for k,v in score(r['raw'],q).items())
   m=dict(correct=sum(r['correct'] for r in rows),by_category={c:dict(correct=sum(r['correct'] for r in rows if r['category']==c)) for c in {q['category'] for q in qs.values()}})
   expected=selection['validation'][name] if split=='validation' else summary['tests'][name]
   assert m['correct']==expected['correct']
   for c in m['by_category']:assert m['by_category'][c]['correct']==expected['by_category'][c]['correct']
   if split=='validation':validation[name]=m
   outputs[name+'_'+split]=rows;count+=len(rows)
 assert max(protocol['candidates'],key=lambda n:selection_key(validation[n]))==selection['winner']
 for name in dict.fromkeys(['v15',selection['winner']]):
  rows=read(ROOT/(name+'_old.json'));assert len(rows)==240 and len({(r['id'],r['shift']) for r in rows})==240
  assert metrics(rows,old_questions())==summary['tests'][name]['old'];outputs[name+'_old']=rows;count+=len(rows)
 assert sha(ROOT/'adapter/adapter_model.safetensors')==summary['adapter_sha256']
 tensors={}
 for path in ROOT.glob('*/adapter/adapter_model.safetensors'):
  w=load_file(str(path));assert len(w)==392 and all(torch.isfinite(x).all() for x in w.values());tensors[str(path.relative_to(ROOT))]=len(w)
 w=load_file(str(ROOT/'adapter/adapter_model.safetensors'));assert len(w)==392 and all(torch.isfinite(x).all() for x in w.values())
 manifest=read(ROOT/'manifest.json')
 for r in manifest:
  p=ROOT/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
 archive=ROOT.with_suffix('.zip')
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  assert hashlib.sha256(z.read('adapter/adapter_model.safetensors')).hexdigest()==summary['adapter_sha256']
 result=dict(summary=summary,outputs=outputs,training_rows=read(ROOT/'train.json'),verification=dict(responses=count,finite_tensors=392,checked_candidate_adapters=tensors,manifest_files=len(manifest),archive_bytes=archive.stat().st_size,archive_sha256=sha(archive)))
 save(ROOT/'verified_results.json',result);print('V17 VERIFIED',result['verification'],flush=True)

if __name__=='__main__':main()
