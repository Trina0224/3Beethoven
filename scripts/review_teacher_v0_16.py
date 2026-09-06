"""Teacher-only structural review of explicitly evaluated scale squares."""
from formulation_grader import parse_expression,shape
from stats_curriculum_v0_16 import score

def teacher_score(raw,q):
    g=score(raw,q)
    if g['math_correct'] is None and g['executable'] and g.get('computed')==q['answer']:
        a=q['bindings']['scale']
        alternative=q['expression'].replace(a+'**2',str(int(a)**2),1)
        if alternative!=q['expression'] and shape(parse_expression(g['normalized_expression'],{}))==shape(parse_expression(alternative,{})):
            g.update(math_correct=True,correct=True,review_required=False,reason='Teacher-only review: scale square evaluated correctly; remaining substituted AST matches exactly. No teacher output repair.')
    return g

def main():
    from collections import Counter
    from prepare_stats_v0_16 import ROOT
    from stats_curriculum_v0_16 import build
    from flight_run_stats_v0_3 import read_json,save_json
    data=build();records=read_json(ROOT/'records.json');reviewed={};counts={}
    for split,rs in records.items():
        qs={q['id']:q for q in data[split]};reviewed[split]=[];tasks=Counter()
        for r in rs:
            grades=[teacher_score(a['raw'],qs[r['id']]) for a in r['attempts']]
            accepted=any(g['correct'] for g in grades)
            reviewed[split].append(dict(id=r['id'],original_accepted=r['accepted'],accepted=accepted,grades=grades))
            if accepted:tasks[qs[r['id']]['task']]+=1
        counts[split]=dict(accepted=sum(tasks.values()),by_task=dict(tasks))
    save_json(ROOT/'structural_review.json',reviewed)
    summary=read_json(ROOT/'summary.json');summary['teacher_only_structural_review']=counts
    save_json(ROOT/'summary.json',summary)
    assert counts['train']['accepted']>=180 and min(counts['train']['by_task'].values())>=42
    assert counts['validation']['accepted']>=30
    print('V16 TEACHER REVIEW',counts,flush=True)

if __name__=='__main__':main()
