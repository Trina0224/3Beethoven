"""Recover nominally shared historical MC results without inventing strict comparability."""
import json
from pathlib import Path
from stats_holdout_v1 import questions
from run_stats_rotation_v1 import rotate
from stats_v0_3_common import parse_answer
D=Path(__file__).resolve().parents[1]/'docs'
qs=questions();expected={(q['id'],s):rotate(q,s)['answer_letter'] for q in qs for s in range(4)}
records=[]
for version,key in [(5,'v05_old'),(9,'v09_old'),(14,'v14_old'),(15,'v15_old'),(16,'v16_old'),(17,'train_8_old')]:
 filename=f'STATS_V0_{version}_RESULTS.json';x=json.loads((D/filename).read_text());rows=x['outputs'][key] if version==17 else x[key]
 assert len(rows)==240 and {(r['id'],r['shift']) for r in rows}==set(expected)
 for r in rows:assert r['expected']==expected[(r['id'],r['shift'])] and r['correct']==(parse_answer(r['raw'])==r['expected'])
 records.append(dict(version=version,source=filename,key=key,correct=sum(r['correct'] for r in rows),all_four_correct=sum(all(r['correct'] for r in rows if r['id']==q['id']) for q in qs),question_ids_and_gold_verified=True,raw_regraded=True,logged_exact_prompt_hash=False,environment=x.get('environment'),comparability='Historical nominal common-test retention trace, not a controlled common-runtime rerun or a blinded learning curve.'))
(D/'STATS_PERMANENT_MC_HISTORY_AUDIT.json').write_text(json.dumps(records,indent=2)+'\n')
print([(r['version'],r['correct'],r['all_four_correct']) for r in records])
