"""Supplemental reviewed equivalences; preserve frozen automatic scores."""
import hashlib
import json
from pathlib import Path
from formulation_grader import parse_expression
from exact_calculator import calculate
import ast


def main():
    root=Path(__file__).resolve().parents[1]/'docs'
    data=json.loads((root/'STATS_V0_14_RESULTS.json').read_text())
    qs={q['id']:q for q in json.loads((root/'STATS_V0_14_FROZEN_QUESTIONS.json').read_text())['test']}
    approvals={
        'baseline':{f'v14_test_binomial_{i:03d}':'Comb denotes the binomial coefficient; complementary percentages and n-r have been correctly evaluated. The numerical binomial probability formula is complete.' for i in range(8)},
        'v13':{
            'v14_test_exactly_one_000':'Both miss probabilities are 41%; 41/100 * 59/100 appears twice, the two mutually exclusive detection cases.',
            'v14_test_at_least_one_006':'Sum of both-detect, only-A-detect and only-B-detect events is equivalent to one minus both-miss.',
            'v14_test_at_least_one_007':'Sum of both-detect, only-A-detect and only-B-detect events is equivalent to one minus both-miss.'},
        'v14':{
            'v14_test_poisson_time_000':'41*162/60 equals rate times duration in minutes by multiplication/division associativity.',
            'v14_test_poisson_time_003':'48*278/60 equals rate times duration in minutes by multiplication/division associativity.',
            'v14_test_binomial_007':'71/100 is the correct complementary probability 1-29/100; the rest of the binomial expression is complete.'}}
    results={}
    for model in ('baseline','v13','v14'):
        rows=[]
        for r in data[model+'_formulation']:
            reason=approvals[model].get(r['id'])
            if reason:
                assert r['review_required'] and not r['correct']
                # Numerical validation supplements, but never replaces, the
                # explicit mathematical justification above.
                expr=r['raw'].removeprefix('Expression:').strip().replace('Comb(', 'comb(')
                assert calculate(ast.unparse(parse_expression(expr,{})))==qs[r['id']]['answer']
            rows.append(dict(id=r['id'],category=r['category'],raw=r['raw'],raw_sha256=hashlib.sha256(r['raw'].encode()).hexdigest(),
                automatic_correct=r['correct'],reviewed_correct=r['correct'] or bool(reason),additional_credit_reason=reason))
        results[model]=dict(automatic_correct=sum(r['automatic_correct'] for r in rows),reviewed_correct=sum(r['reviewed_correct'] for r in rows),n=len(rows),
            by_category={c:sum(r['reviewed_correct'] for r in rows if r['category']==c) for c in sorted({r['category'] for r in rows})},rows=rows)
    output=dict(policy='Post-hoc semantic review of final fully substituted formulas. Original automatic scores, prompts, checkpoints and raw answers are unchanged. No credit for unresolved X or a correct earlier setup followed by a wrong final calculation.',models=results)
    (root/'STATS_V0_14_SEMANTIC_REVIEW.json').write_text(json.dumps(output,indent=2)+'\n')
    print({k:{j:v[j] for j in ('automatic_correct','reviewed_correct','n','by_category')} for k,v in results.items()})


if __name__=='__main__':main()
