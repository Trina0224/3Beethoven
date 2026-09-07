"""List all unresolved mathematical outputs, including validation and historical tests."""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
d=json.loads((R/'docs/STATS_SEED_REPLICATION_RESULTS.json').read_text())
qs={}
for p in (R/'docs').glob('STATS_V0_*_FROZEN_QUESTIONS.json'):
 obj=json.loads(p.read_text())
 for rows in obj.values():
  if isinstance(rows,list):
   for q in rows:
    if isinstance(q,dict) and 'id' in q:qs[q['id']]=q
for q in json.loads((R/'docs/STATS_V19_PERTURBATION_QUESTIONS.json').read_text()):qs[q['id']]=q
unique={}
for filename,rows in d.items():
 if not isinstance(rows,list):continue
 for row in rows:
  if not isinstance(row,dict) or not row.get('review_required') or 'raw' not in row:continue
  run=filename.split('/')[0];key=(run,row['id'],row['raw'])
  if key not in unique:
   q=qs.get(row['id'],{})
   unique[key]=dict(run=run,id=row['id'],raw=row['raw'],reason=row.get('reason'),normalized_expression=row.get('normalized_expression'),question=q.get('question'),reference=q.get('expression'),files=[])
  unique[key]['files'].append(filename)
(R/'docs/STATS_SEED_REPLICATION_PENDING_AUDIT.json').write_text(json.dumps(list(unique.values()),ensure_ascii=False,indent=2)+'\n')
print('Unique pending outputs',len(unique))
for i,x in enumerate(unique.values()):print(i,json.dumps(x,ensure_ascii=False))
