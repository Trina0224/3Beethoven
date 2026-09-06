"""Summarize paired frozen outcomes without selecting on the test."""
from run_stats_v0_17 import ROOT,read,save

def main():
 s=read(ROOT/'summary.json');base=read(ROOT/'v15_test.json');b={r['id']:r for r in base};out={}
 for name in s['tests']:
  rows=read(ROOT/(name+'_test.json'))
  improved=[r['id'] for r in rows if r['correct'] and not b[r['id']]['correct']]
  lost=[r['id'] for r in rows if not r['correct'] and b[r['id']]['correct']]
  retained=[r['id'] for r in rows if r['correct'] and b[r['id']]['correct']]
  non_target_lost=[i for i in lost if b[i]['category'] not in ('moment','poisson_scaled')]
  out[name]=dict(correct=sum(r['correct'] for r in rows),newly_correct=improved,newly_wrong=lost,correct_in_both=retained,
      non_target_losses=non_target_lost,improvement_gate=len(improved)-len(lost)>=6,retention_gate=len(non_target_lost)<=3,
      pending=[r['id'] for r in rows if r['review_required']],truncated=[r['id'] for r in rows if r['hit_token_limit']])
 save(ROOT/'paired_outcomes.json',out)
 print('V17 PAIRED', {n:{k:len(v) if isinstance(v,list) else v for k,v in r.items()} for n,r in out.items()})

if __name__=='__main__':main()
