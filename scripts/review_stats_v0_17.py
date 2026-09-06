"""Explicit supplementary semantic review; never rewrites frozen scores/selection."""
import hashlib
from pathlib import Path
from run_stats_v0_17 import ROOT,read,save
from stats_curriculum_v0_17 import build,selection_key

def main():
 repo=Path(__file__).resolve().parents[1];credits=read(repo/'docs/STATS_V0_17_REVIEW_CREDITS.json')
 summary=read(ROOT/'summary.json');data=build();out={};used=set();all_rows={}
 names={'validation':summary['protocol']['candidates'],'test':list(summary['tests'])}
 for split,models in names.items():
  for model in models:
   name=model+'_'+split;rows=read(ROOT/(name+'.json'));assert len(rows)==len(data[split]);decisions=[]
   for r in rows:
    key=name+'::'+r['id'];credit=credits.get(key)
    if credit:
     assert r['raw']==credit['raw'] and not r['correct'],key;used.add(key)
    decisions.append(dict(id=r['id'],category=r['category'],raw_sha256=hashlib.sha256(r['raw'].encode()).hexdigest(),
      automatic_correct=r['correct'],reviewed_correct=r['correct'] or bool(credit),
      reason=credit['reason'] if credit else 'Full expression reviewed against question and reference; frozen decision retained.'))
   all_rows[name]=decisions
   out[name]=dict(n=len(rows),automatic_correct=sum(r['correct'] for r in rows),correct=sum(r['reviewed_correct'] for r in decisions),
      by_category={c:dict(n=sum(r['category']==c for r in rows),correct=sum(r['reviewed_correct'] for r in decisions if r['category']==c)) for c in sorted({r['category'] for r in rows})},
      additional_credits=[r['id'] for r in decisions if r['reviewed_correct'] and not r['automatic_correct']],decisions=decisions)
 assert used==set(credits)
 validation={n:out[n+'_validation'] for n in names['validation']}
 supplemental_winner=max(names['validation'],key=lambda n:selection_key(validation[n]))
 base={r['id']:r for r in all_rows['v15_test']};paired={}
 for name in names['test']:
  rows=all_rows[name+'_test'];new=[r['id'] for r in rows if r['reviewed_correct'] and not base[r['id']]['reviewed_correct']]
  lost=[r['id'] for r in rows if not r['reviewed_correct'] and base[r['id']]['reviewed_correct']]
  non_target=[i for i in lost if base[i]['category'] not in ('moment','poisson_scaled')]
  paired[name]=dict(newly_correct=new,newly_wrong=lost,non_target_losses=non_target,net_gain=len(new)-len(lost),improvement_gate=len(new)-len(lost)>=6,non_target_retention_gate=len(non_target)<=3)
 result=dict(method='Full response-level semantic review against requested quantity and reference; explicit raw-bound exceptions only. Frozen automatic scores and original validation selection remain unchanged.',
    response_count=sum(r['n'] for r in out.values()),models=out,paired=paired,supplemental_validation_winner=supplemental_winner,
    original_winner=summary['selection']['winner'],selection_unchanged=supplemental_winner==summary['selection']['winner'])
 save(ROOT/'semantic_review.json',result)
 print('V17 REVIEW',{n:r['correct'] for n,r in out.items()},'selection unchanged',result['selection_unchanged'])

if __name__=='__main__':main()
