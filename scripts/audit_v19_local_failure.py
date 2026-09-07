"""Describe observed output errors, without claiming access to internal mechanisms."""
import json
from pathlib import Path
from fractions import Fraction as F
from exact_calculator import calculate
D=Path(__file__).resolve().parents[1]/'docs';x=json.loads((D/'STATS_V0_19_RESULTS.json').read_text())
qmap={q['id']:q for q in x['frozen_questions.json']['old_test']};decisions=[]
for r in x['outputs']['epoch_1_old_test.json']:
 if r['category']!='exactly_one':continue
 q=qmap[r['id']];a,b=map(F,(q['bindings']['miss_a'],q['bindings']['miss_b']))
 value=F(calculate(r['raw'].split('Expression:',1)[1].strip()))
 label='correct' if value==(1-a)*b+a*(1-b) else ('only_A_detects_B_misses' if value==(1-a)*b else ('both_miss' if value==a*b else 'other'))
 decisions.append(dict(id=r['id'],raw=r['raw'],reference=q['expression'],classification=label))
refs={}
for v in (13,14,15):
 for split,rows in json.loads((D/f'STATS_V0_{v}_FROZEN_QUESTIONS.json').read_text()).items():
  for q in rows:refs[q['id']]=(split,q)
replay=[]
for r in x['frozen_questions.json']['train']:
 if r['category']!='exactly_one':continue
 split,q=refs[r['source_id']];assert split=='train'
 expr=r['target'].split('Expression:',1)[1].strip()
 assert calculate(expr)==q['answer']
 replay.append(dict(source_id=r['source_id'],target=r['target'],reference=q['expression'],training_only=True,exact_value_verified=True))
assert len(replay)==24
result=dict(test_decisions=decisions,counts={k:sum(r['classification']==k for r in decisions) for k in {r['classification'] for r in decisions}},replay_audit=replay,interpretation='Eight outputs omit the B-detects/A-misses branch; three compute both misses; one is correct. All 24 replay targets have correct numerical references and training-only source IDs. This local error signature does not establish erased internal knowledge or identify the causal training mechanism.')
(D/'STATS_V19_LOCAL_FAILURE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(result['counts'])
