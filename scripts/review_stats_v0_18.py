"""Apply explicit raw-bound semantic credits without changing training decisions."""
import copy,hashlib
from run_stats_v0_18 import ROOT,metrics
from flight_run_stats_v0_3 import read_json as read,save_json as save

def main():
    result=read(ROOT/'verified_results.json');credits=read(ROOT/'review_credits.json',[])
    outputs=copy.deepcopy(result['outputs']);seen=set()
    questions={q['id']:q for rows in read(ROOT/'frozen_questions.json').values() for q in rows}
    for c in credits:
        key=(c['file'],c['id']);assert key not in seen;seen.add(key)
        rows=[r for r in outputs[c['file']] if r['id']==c['id']];assert len(rows)==1;r=rows[0]
        assert hashlib.sha256(r['raw'].encode()).hexdigest()==c['raw_sha256']
        assert not r['correct'] and r['review_required'] and r['computed']==questions[r['id']]['answer']
        r.update(correct=True,review_required=False,semantic_credit=c['reason'])
    review=dict(credits=credits,metrics={p:metrics(rows) for p,rows in outputs.items()},
                note='Supplemental semantic review only. Original responses, automatic scores, stopping decisions and test candidates are immutable.')
    names=result['summary']['tests'];baseline={r['id']:r for r in outputs['v15_test.json']};paired={}
    for name in names:
        rows=outputs[name+'_test.json'];paired[name]=dict(newly_correct=sum(r['correct'] and not baseline[r['id']]['correct'] for r in rows),newly_wrong=sum(not r['correct'] and baseline[r['id']]['correct'] for r in rows))
    review['paired_vs_v15']=paired
    a={r['id']:r for r in outputs[f'control_{len(result["summary"]["stages"])}_test.json']}
    b=outputs[f'stage_{len(result["summary"]["stages"])}_test.json']
    review['curriculum_vs_control']=dict(newly_correct=sum(r['correct'] and not a[r['id']]['correct'] for r in b),newly_wrong=sum(not r['correct'] and a[r['id']]['correct'] for r in b))
    save(ROOT/'semantic_review.json',review);print('V18 SEMANTIC REVIEW',len(credits),flush=True)

if __name__=='__main__':main()
