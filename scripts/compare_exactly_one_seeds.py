"""Classify saved event expressions; no inference or training."""
import json,re,hashlib
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
from exact_calculator import calculate
D=Path(__file__).resolve().parents[1]/'docs'
data=json.loads((D/'STATS_SEED_REPLICATION_RESULTS.json').read_text())
old=json.loads((D/'STATS_V0_19_FROZEN_QUESTIONS.json').read_text())
qs={q['id']:q for q in old['old_test']}
prior=json.loads((D/'STATS_V19_ANALYSIS_RESULTS.json').read_text())
out=[]
def classify(run,split,rows):
 for row in rows:
  if row['category']!='exactly_one':continue
  qid=re.search(r'v19_old_test_exactly_one_\d{3}',row['id']).group()
  q=qs[qid];a,b=map(F,(q['bindings']['miss_a'],q['bindings']['miss_b']))
  expr=row['raw'].split('Expression:',1)[1].strip();v=F(calculate(expr))
  signatures={'correct':(1-a)*b+a*(1-b),'only_A_detects':(1-a)*b,'both_miss':a*b,'same_outcome_XNOR':a*b+(1-a)*(1-b),'B_misses_regardless_A':b,'A_detects_regardless_B':1-a}
  matches=[k for k,val in signatures.items() if v==val]
  assert len(matches)<=1,(row['id'],matches)
  label=matches[0] if matches else 'other'
  binding_errors={('2027','v19_old_test_exactly_one_001'):('31/100','4/5'),('31415','v19_old_test_exactly_one_001'):('31/100','4/13'),('31415','v19_old_test_exactly_one_005'):('65/100,40/100','13/50,9/25'),('31415','v19_old_test_exactly_one_007'):('35/100','11/20')}
  detail=binding_errors.get((run,row['id']))
  if detail:
   assert label=='other';label='XOR_structure_wrong_binding'
  out.append(dict(run=run,split=split,id=row['id'],base_id=qid,raw=row['raw'],reference=q['expression'],classification=label,wrong_binding=detail,raw_sha256=hashlib.sha256(row['raw'].encode()).hexdigest()))
for seed in (1919,2027,31415):
 for split in ('anchor_old_test','perturbation'):classify(str(seed),split,data[f'v19_seed_{seed}/{split}.json'])
classify('original_v15','anchor_old_test',data['v19_seed_1919/v15_old_test.json'])
classify('original_v15','perturbation',prior['v15_formula.json'])
counts={run:{split:dict(Counter(r['classification'] for r in out if r['run']==run and r['split']==split)) for split in ('anchor_old_test','perturbation')} for run in ('1919','2027','31415','original_v15')}
result=dict(counts=counts,rows=out,method='Exact rational event signatures plus inspected XOR binding errors; descriptive output patterns, not proof of causal mechanism. Existing saved outputs only.')
(D/'STATS_SEED_EXACTLY_ONE_COMPARISON.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(counts,indent=2))
