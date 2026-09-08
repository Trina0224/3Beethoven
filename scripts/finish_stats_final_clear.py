"""Review existing outputs and package the one final comparison without retraining."""
import argparse,json,zipfile,hashlib,copy
from pathlib import Path
from collections import Counter
from review_stats_semantic_course import prove
from stats_consolidation_pilot import digest
ROOT=Path(__file__).resolve().parents[1]
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--package',action='store_true');a=p.parse_args();root=a.root
 data=json.loads((ROOT/'docs/STATS_FINAL_CLEAR_DATA.json').read_text());manual_path=root/'manual_reviews.json';manual=json.loads(manual_path.read_text()) if manual_path.exists() else []
 manual={(r['question_sha256'],r['raw_sha256']):r for r in manual};summary={};pending=[];proofs=[];allrows={}
 for model in ['v15','student']:
  summary[model]={}
  for suite in ['development','test']:
   path=root/'evaluation'/model/(suite+'.json')
   if not path.exists():continue
   rows=json.loads(path.read_text());questions={q['id']:q for q in data[suite]}
   if len(rows)!=len(questions):continue
   for row in rows:
    q=questions[row['id']];assert row['question_sha256']==digest(q)
    if not row['review_required']:continue
    key=(row['question_sha256'],row['raw_sha256'])
    if key in manual:correct=manual[key]['correct'];proof=manual[key]['reason']
    else:
     try:proof=prove(row,q);correct=True
     except (ValueError,AssertionError,KeyError,TypeError,SyntaxError,ZeroDivisionError):pending.append(dict(model=model,suite=suite,question=q,row=row));continue
    row.update(correct=correct,primary_correct=correct,math_correct=correct,review_required=False,reason=str(proof))
    proofs.append(dict(model=model,suite=suite,id=row['id'],question_sha256=row['question_sha256'],raw_sha256=row['raw_sha256'],correct=correct,proof=proof))
   save(path.with_name(suite+'_reviewed.json'),rows);allrows[(model,suite)]=rows
   summary[model][suite]=dict(n=len(rows),correct=sum(r['primary_correct'] for r in rows),pending=sum(r['review_required'] for r in rows),categories={cat:dict(n=sum(r['category']==cat for r in rows),correct=sum(r['category']==cat and r['primary_correct'] for r in rows)) for cat in data['manifest']['categories']})
 result=dict(models=summary,pending=len(pending),semantic_reviews=len(proofs),status='incomplete',success=False)
 if all('test' in summary[m] and 'development' in summary[m] for m in ['v15','student']):
  b=summary['v15']['test'];s=summary['student']['test'];changes={cat:s['categories'][cat]['correct']-b['categories'][cat]['correct'] for cat in data['manifest']['categories']}
  before={r['id']:r for r in allrows[('v15','test')]};pairs=Counter();regressions=[]
  for r in allrows[('student','test')]:
   x=before[r['id']];pairs[str(int(x['primary_correct']))+'->'+str(int(r['primary_correct']))]+=1
   if x['primary_correct'] and not r['primary_correct']:regressions.append(r['id'])
  result.update(status='complete' if not pending else 'review_pending',gain=s['correct']-b['correct'],category_changes=changes,pairs=dict(pairs),regression_ids=regressions,success=not pending and s['correct']-b['correct']>=5 and all(v>=0 for v in changes.values()))
 save(root/'summary.json',result);save(root/'pending_reviews.json',pending);save(root/'semantic_reviews.json',proofs)
 print('FINAL_CLEAR_REVIEW',json.dumps(result),flush=True)
 if a.package:
  assert result['status']=='complete'
  assert (root/'student/training_complete.json').exists()
  out=root.parent/'3beethoven_final_clear_student.zip'
  with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
   for item in sorted(root.rglob('*')):
    if not item.is_file() or 'checkpoints' in item.parts or '__pycache__' in item.parts:continue
    if item.is_relative_to(ROOT) and item.suffix not in ('.py','.json','.md'):continue
    z.write(item,str(item.relative_to(root)))
  receipt=dict(path=str(out),bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),success=result['success'])
  save(root/'delivery.json',receipt);print('FINAL_CLEAR_DELIVERABLE',json.dumps(receipt),flush=True)
if __name__=='__main__':main()
