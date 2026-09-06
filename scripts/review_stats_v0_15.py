"""Preserve frozen automatic scores; add two justified final-expression equivalences."""
import ast
import hashlib
import json
from pathlib import Path
from exact_calculator import calculate
from formulation_grader import parse_expression


def main():
    root=Path(__file__).resolve().parents[1]/'docs'
    data=json.loads((root/'STATS_V0_15_RESULTS.json').read_text())
    qs={q['id']:q for q in json.loads((root/'STATS_V0_15_FROZEN_QUESTIONS.json').read_text())['test']}
    approvals={
      'v15_test_poisson_time_000':'188*(50/60)=50*(188/60) by multiplication/division associativity. Both the event rate and seconds-to-minutes conversion are correctly included.',
      'v15_test_uniform_time_004':'The multiplier (385/60-385/60+1) is identically one, so the expression reduces to (48+385/60)/2, the conditional total wait. This is a symbolic cancellation, not merely numerical coincidence.'}
    models={}
    for model in ('baseline','v14','v15'):
        rows=[]
        for r in data[model+'_formulation']:
            reason=approvals.get(r['id']) if model=='v15' else None
            if reason:
                assert r['review_required'] and not r['correct']
                node=parse_expression(r['raw'].removeprefix('Expression:').strip(),{})
                assert calculate(ast.unparse(node))==qs[r['id']]['answer']
            rows.append(dict(id=r['id'],category=r['category'],raw=r['raw'],raw_sha256=hashlib.sha256(r['raw'].encode()).hexdigest(),automatic_correct=r['correct'],reviewed_correct=r['correct'] or bool(reason),additional_credit_reason=reason))
        models[model]=dict(n=len(rows),automatic_correct=sum(r['automatic_correct'] for r in rows),reviewed_correct=sum(r['reviewed_correct'] for r in rows),by_category={c:sum(r['reviewed_correct'] for r in rows if r['category']==c) for c in sorted({r['category'] for r in rows})},rows=rows)
    output=dict(policy='Supplemental final-formulation review. Frozen scores and all raw answers remain unchanged. Unresolved X, missing variance terms and wrong final expressions earn no credit.',models=models)
    (root/'STATS_V0_15_SEMANTIC_REVIEW.json').write_text(json.dumps(output,indent=2)+'\n')
    print({m:{k:v[k] for k in ('automatic_correct','reviewed_correct','by_category')} for m,v in models.items()})


if __name__=='__main__':main()
