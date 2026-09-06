"""Explicit full-output semantic review; preserves the automatic diagnostic grades."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

def review(data):
    rows=[];counts=defaultdict(lambda:dict(n=0,correct=0))
    for model,answers in data['models'].items():
        for r in answers:
            stage,family=r['stage'],r['family']
            index=int(r['story_id'].rsplit('_',1)[1])
            hit=r['math_correct'] is True
            reason='Reviewed full output: correct final substituted formula.' if hit else 'No complete correct requested answer; unresolved X, wrong formula/binding, or unfinished response.'
            if stage.startswith('extract_'):
                if stage=='extract_offset':hit=True
                elif stage=='extract_variance':hit=not(model=='v15' and family=='poisson_scaled' and index==3)
                elif stage=='extract_mean':hit=not(model=='v15' and family=='moment' and index==7)
                elif stage=='extract_scale':hit=model=='v15' and family=='poisson_scaled' and index in (0,2,3,6)
                reason=('Correct requested scalar is explicitly stated in the response, including prose.' if hit else
                        'Wrong or unfinished multiplier/value; do not credit a truncated trailing digit or an intermediate correct number.')
            if stage=='mean' and family=='poisson_scaled':
                hit=index in ((1,4,5,6,7) if model=='v14' else (1,3,6,7))
                if hit:reason='Correct fully substituted mean formula appears, followed by consistent arithmetic/final answer; prose formatting does not remove credit.'
            if model=='v14' and family=='poisson_scaled' and stage=='variance' and index==2:
                hit=True;reason='Explicit 3^2 * 55 in prose and consistent final variance 495; no later contradictory conclusion.'
            flags=[]
            if model=='v15' and family=='moment' and stage=='extract_mean' and index==6:
                flags.append('Correct E[X]=60, but additionally confuses it with the transformed quantity expectation. Binding credit only.')
            if model=='v15' and family=='moment' and stage=='mean' and index==0:
                flags.append('Correct final mean formula; earlier arithmetic and variance claims are wrong. Final formulation credit only.')
            row=dict(model=model,id=r['id'],family=family,stage=stage,raw_sha256=hashlib.sha256(r['raw'].encode()).hexdigest(),
                     automatic=r['math_correct'],semantic_correct=hit,reason=reason,flags=flags)
            rows.append(row)
            key=(model,family,stage);counts[key]['n']+=1;counts[key]['correct']+=hit
    summary={m:{f:{s:c for (mm,ff,s),c in counts.items() if mm==m and ff==f}
        for f in ('moment','poisson_scaled')} for m in data['models']}
    return dict(policy='Full-output semantic review: explicit correct scalar for extraction; correct fully substituted formula with consistent continuation for formulation; no gold repair. Original auto grades unchanged.',summary=summary,rows=rows)

if __name__=='__main__':
    docs=Path(__file__).resolve().parents[1]/'docs'
    source=(docs/'STATS_V0_15_DIAGNOSTIC_RESULTS.json').read_bytes()
    assert hashlib.sha256(source).hexdigest()=='6af9cf1e1a300785da3352557630095cb0e08c2985219ece072c41417e4f8de2', 'Review applies only to inspected responses'
    data=json.loads(source)
    result=review(data)
    (docs/'STATS_V0_15_DIAGNOSTIC_REVIEW.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2))
