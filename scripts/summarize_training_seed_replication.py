"""Verify every fixed-anchor answer and report training-run variation separately."""
import json,statistics,hashlib,base64,gzip
from pathlib import Path
from stats_curriculum_v0_19 import build,digest,prompt,score
from stats_v0_3_common import parse_answer
REPO=Path(__file__).resolve().parents[1]
def main():
 data=json.loads((REPO/'docs/STATS_SEED_REPLICATION_RESULTS.json').read_text())
 reviewpath=REPO/'docs/STATS_SEED_REPLICATION_SEMANTIC_REVIEW.json'
 reviews=json.loads(reviewpath.read_text()) if reviewpath.exists() else []
 credits={(r['run'],r['file'],r['id']):r for r in reviews}
 anchor=json.loads((REPO/'docs/STATS_PERMANENT_ANCHOR_V1.json').read_text())
 probes=json.loads((REPO/'docs/STATS_V19_PERTURBATION_QUESTIONS.json').read_text())
 frozen=build();results=[];pending=[];count=0
 def reviewed(run,file,r):
  key=(run,file,r['id']);judgment=credits.get(key)
  if r.get('review_required') and not judgment:pending.append(dict(run=run,file=file,**r))
  return bool(r['correct'] or (judgment and judgment['credit']))
 for family in ('v15','v19'):
  for seed in (1919,2027,31415):
   run=f'{family}_seed_{seed}';out={'family':family,'seed':seed,'automatic':{},'reviewed':{},'by_family':{},'by_variant':{}}
   completion=data[run+'/replication_complete.json'];out['training']=completion['training'];out['adapter_sha256']=completion['adapter_sha256']
   out['selection']=data.get(run+'/selection.json',{'rule':'validation loss','best_checkpoint':completion['training'].get('best_checkpoint')})
   for name,qs in [('anchor_old_test',frozen['old_test']),('anchor_new_test',frozen['new_test']),('perturbation',probes)]:
    rows=data[f'{run}/{name}.json'];lookup={q['id']:q for q in qs};assert len(rows)==len(qs)==len({r['id'] for r in rows})
    for r in rows:
     q=lookup[r['id']];assert r['question_sha256']==digest(q) and r['prompt']==prompt(q)
     assert r['correct']==score(r['raw'],q)['correct'];count+=1
    vals={r['id']:reviewed(run,name,r) for r in rows}
    out['automatic'][name]=sum(r['correct'] for r in rows);out['reviewed'][name]=sum(vals.values())
    out['by_family'][name]={c:sum(vals[r['id']] for r in rows if r['category']==c) for c in sorted({r['category'] for r in rows})}
    if name=='perturbation':
     out['by_variant']={v:sum(vals[r['id']] for r in rows if lookup[r['id']]['variant']==v) for v in sorted({q['variant'] for q in qs})}
   rows=data[run+'/anchor_mc.json'];expected={(r['id'],r['shift']):r['expected'] for r in anchor['mc']['rotations']}
   assert len(rows)==240==len({(r['id'],r['shift']) for r in rows})
   for r in rows:assert r['expected']==expected[(r['id'],r['shift'])] and r['correct']==(parse_answer(r['raw'])==r['expected'])
   count+=240;out['automatic']['mc']=out['reviewed']['mc']=sum(r['correct'] for r in rows)
   out['reviewed']['mc_all_four']=sum(all(r['correct'] for r in rows if r['id']==i) for i in {r['id'] for r in rows})
   results.append(out)
 def stats(xs):return dict(values=xs,mean=statistics.mean(xs),sample_sd=statistics.stdev(xs),min=min(xs),max=max(xs))
 aggregates={family:{'scores':{k:stats([r['reviewed'][k] for r in results if r['family']==family]) for k in results[0]['reviewed']},'variants':{k:stats([r['by_variant'][k] for r in results if r['family']==family]) for k in results[0]['by_variant']}} for family in ('v15','v19')}
 summary=dict(runs=results,aggregates=aggregates,verified_anchor_responses=count,pending=pending,review_count=len(reviews),scope='Three retrainings conditional on each fixed parent; no between-parent replication; exposed benchmarks; SD across runs, not item sampling SE.')
 (REPO/'docs/STATS_SEED_REPLICATION_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 (REPO/'docs/STATS_SEED_REPLICATION_RESULTS.json.gz.b64').write_text(base64.b64encode(gzip.compress(json.dumps(data).encode())).decode()+'\n')
 print(json.dumps(dict(verified=count,aggregates=aggregates,pending=len(pending)),ensure_ascii=False))
if __name__=='__main__':main()
