"""Read-only summaries and compact exports for the bounded repair."""
import argparse,json,hashlib
from pathlib import Path
from collections import defaultdict

def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--input',type=Path,required=True);a=p.parse_args()
 previous=a.input/'3beethoven_first_students'
 out={'controls':{},'repairs':{},'pending':[],'teacher_calls':0,'independent_test_executed':False}
 for name in ('v15','step24','step33'):
  out['controls'][name]={}
  prior=previous/'v15_selection_rows' if name=='v15' else previous/'original_v15_continued_lora_seed_31415'/name
  for suite in ('old','chain','event','replay'):
   f=a.root/'control'/name/f'{suite}.json'
   if not f.exists():continue
   rows=read(f);original={r['id']:r for r in read(prior/f'{suite}.json')} if suite!='replay' else {}
   by=defaultdict(lambda:{'n':0,'correct':0,'pending':0});same=0
   for r in rows:
    known=original.get(r['id'])
    if known and (known['raw_sha256'],known['question_sha256'])==(r['raw_sha256'],r['question_sha256']):
     same+=1
     for k in ('primary_correct','review_required'):r[k]=known[k]
    by[r['category']]['n']+=1;by[r['category']]['correct']+=int(r['primary_correct']);by[r['category']]['pending']+=int(r['review_required'])
    if r['review_required']:out['pending'].append({'stage':name,'suite':suite,**r})
   out['controls'][name][suite]={'n':len(rows),'correct':sum(r['primary_correct'] for r in rows),'same_raw':same if original else None,'categories':dict(by)}
 for seed in (2027,31415):
  folder=a.root/f'low_lr/original_v15_continued_lora_seed_{seed}'
  if not folder.exists():continue
  history=read(folder/'validation_history.json') if (folder/'validation_history.json').exists() else []
  out['repairs'][seed]={'history':history,'complete':read(folder/'training_complete.json') if (folder/'training_complete.json').exists() else None}
  for h in history:
   for suite in ('old','chain','event'):
    for r in read(folder/f"step_{h['step']}"/f'{suite}.json'):
     if r['review_required']:out['pending'].append({'stage':f"repair_{seed}_{h['step']}",'suite':suite,**r})
 (a.root/'REPAIR_SUMMARY.json').write_text(json.dumps(out,indent=2)+'\n')
 brief={'controls':out['controls'],'repairs':{k:{'history':[{'step':h['step'],'scores':{s:m['correct'] for s,m in h['metrics'].items()},'failed':[n for n,b in h['gate']['checks'].items() if not b]} for h in v['history']],'complete':v['complete']} for k,v in out['repairs'].items()},'pending':out['pending']}
 print('REPAIR_SUMMARY',json.dumps(brief),flush=True)
if __name__=='__main__':main()
