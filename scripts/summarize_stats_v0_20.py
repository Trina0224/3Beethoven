"""Summarize saved outputs without new model calls or rewriting scores."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/kaggle/working/3beethoven_stats_v0_20')
def read(p):return json.loads(p.read_text())
def stats(rows):
 cats=sorted({r.get('category','mc') for r in rows})
 return dict(n=len(rows),correct=sum(r['correct'] for r in rows),pending=sum(r.get('review_required',False) for r in rows),by_category={c:dict(n=sum(r.get('category','mc')==c for r in rows),correct=sum(r['correct'] for r in rows if r.get('category','mc')==c)) for c in cats})
out={'models':{},'validation_history':{},'pending':[],'validation_pending':[],'event_errors':[],'old_pairs':{},'new_pairs':{},'receipts':{},'status':'automatic; semantic review pending'}
for name in ('parent','seed_2027','seed_31415'):
 folder=ROOT/name;out['models'][name]={}
 for split in ('event_test','old_test','new_test','mc'):
  path=folder/f'{split}.json';rows=read(path);out['models'][name][split]=stats(rows)
  out['receipts'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
  for r in rows:
   if r.get('review_required'):out['pending'].append(dict(model=name,split=split,**r))
   if split=='event_test' and not r['correct']:out['event_errors'].append(dict(model=name,**r))
 if name!='parent':out['validation_history'][name]=read(folder/'selection.json')
 paths=[folder/'old_validation.json'] if name=='parent' else [folder/f'step_{step}'/'old_validation.json' for step in (24,48)]
 for path in paths:
  for r in read(path):
   if r.get('review_required'):out['validation_pending'].append(dict(model=name,split=str(path.relative_to(folder)),**r))
 if name!='parent':
  base={r['id']:r for r in read(ROOT/'parent'/'old_test.json')}
  out['old_pairs'][name]=[dict(id=r['id'],category=r['category'],parent_correct=base[r['id']]['correct'],correct=r['correct'],parent_raw=base[r['id']]['raw'],raw=r['raw']) for r in read(folder/'old_test.json')]
  base={r['id']:r for r in read(ROOT/'parent'/'new_test.json')}
  out['new_pairs'][name]=[dict(id=r['id'],category=r['category'],parent_correct=base[r['id']]['correct'],correct=r['correct'],parent_raw=base[r['id']]['raw'],raw=r['raw']) for r in read(folder/'new_test.json')]
parent=out['models']['parent']
for name in ('seed_2027','seed_31415'):
 m=out['models'][name];selection=out['validation_history'][name]
 out['models'][name]['automatic_gate']=dict(validation_pass=selection['selected_step'] is not None,event_exactly_one=m['event_test']['by_category']['exactly_one']['correct']>=parent['event_test']['by_category']['exactly_one']['correct']+4,event_total=m['event_test']['correct']>=parent['event_test']['correct']+8,old_total=m['old_test']['correct']>=parent['old_test']['correct']-4,old_families=all(m['old_test']['by_category'][c]['correct']>=v['correct']-2 for c,v in parent['old_test']['by_category'].items()),mc=m['mc']['correct']>=parent['mc']['correct']-4)
(ROOT/'review_bundle.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('pending','validation_pending','event_errors','old_pairs','new_pairs')},indent=2))
print('PENDING',len(out['pending']),'EVENT_ERRORS',len(out['event_errors']))
