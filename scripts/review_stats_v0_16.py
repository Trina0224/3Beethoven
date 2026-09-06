"""Supplemental semantic setup review, preserving every automatic score."""
import hashlib
import json
from pathlib import Path
from review_teacher_v0_16 import teacher_score
from formulation_grader import parse_expression,shape
from stats_curriculum_v0_16 import build
from exact_calculator import calculate

def review(data):
    qs={q['id']:q for q in build()['test']};decisions=[];summary={}
    for model in ('v15','v16'):
        summary[model]={}
        for condition in ('unaided','aided'):
            rows=data[model+'_'+condition];judged=[]
            for r in rows:
                q=qs[r['id']];hit=r['correct'];reason='Original automatic grade retained.';flags=[]
                if model=='v16' and q['category']=='moment' and not hit:
                    checked=teacher_score(r['raw'],q)
                    if checked['correct']:
                        hit=True;reason='Scale square is evaluated correctly; remaining substituted formula and exact value verified.'
                if model=='v15' and condition=='unaided' and r['id']=='v16_test_binomial_003':
                    assert '(4 choose 1) * (1/2) ** 1 * (1 - 1/2) ** (4 - 1)' in r['raw']
                    assert calculate('4*(1/2)*(1/2)**3')==q['answer']
                    hit=True;reason='Correct fully substituted binomial setup written in prose, with consistent final answer.'
                    flags.append('Semantic formulation credit; choose notation is not executable by the current parser.')
                if model=='v15' and condition=='aided' and r['id'] in ('v16_test_moment_008','v16_test_moment_010'):
                    expression=r['raw'].splitlines()[1].strip()
                    assert shape(parse_expression(expression,{}))==shape(parse_expression(q['expression'],{}))
                    assert calculate(expression)==q['answer']
                    hit=True;reason='Complete correct numerical formulation explicitly present; subsequent lines evaluate that formula. Project scores formulation separately from arithmetic.'
                    if r['id'].endswith('008'):flags.append('Later addition is wrong by 100; not credited as a correct final numeric answer.')
                row=dict(model=model,condition=condition,id=r['id'],category=r['category'],automatic=r['correct'],
                    semantic_correct=hit,raw_sha256=hashlib.sha256(r['raw'].encode()).hexdigest(),reason=reason,flags=flags)
                decisions.append(row);judged.append(row)
            summary[model][condition]=dict(n=len(judged),automatic=sum(r['correct'] for r in rows),
                semantic_correct=sum(r['semantic_correct'] for r in judged),
                by_category={c:dict(n=sum(r['category']==c for r in judged),correct=sum(r['semantic_correct'] for r in judged if r['category']==c)) for c in sorted({r['category'] for r in judged})})
        summary[model]['retention']=data['summary']['models'][model]['retention']['overall']
    pairs={}
    for condition in ('unaided','aided'):
        a={r['id']:r['semantic_correct'] for r in decisions if r['model']=='v15' and r['condition']==condition}
        b={r['id']:r['semantic_correct'] for r in decisions if r['model']=='v16' and r['condition']==condition}
        pairs[condition]=dict(newly_correct=sum(b[i] and not a[i] for i in a),newly_wrong=sum(a[i] and not b[i] for i in a),both_correct=sum(a[i] and b[i] for i in a))
    return dict(policy='Fully substituted setup is the project endpoint. Preserve auto grades. Arithmetic-only errors after a correct setup are separately flagged; wrong later reformulations are not repaired. No correctness from numerical coincidence alone.',summary=summary,paired=pairs,decisions=decisions)

if __name__=='__main__':
    docs=Path(__file__).resolve().parents[1]/'docs';raw=(docs/'STATS_V0_16_RESULTS.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='b8d8b385484d79a9a3abd88b16ab366fec680adcbbe5ad74ed863129373de5f4'
    result=review(json.loads(raw));(docs/'STATS_V0_16_SEMANTIC_REVIEW.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(summary=result['summary'],paired=result['paired']),indent=2))
