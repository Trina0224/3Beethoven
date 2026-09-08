"""Domain-equivalence review beside, not inside, the unchanged student grader.

Only pending executable answers are eligible. Raw outputs and automatic results
are retained. Unproved pending responses require explicit review by the operator.
"""
import argparse
import ast
import copy
import json
import re
from fractions import Fraction as F
from pathlib import Path
from stats_consolidation_semantics import spec_for, oracle
from stats_consolidation_grader import reviewed_shape
from stats_expansion_teacher import partial_refs
from stats_consolidation_pilot import digest
from run_stats_consolidation_compare import suite_metrics, validation_suites, gate_result


def event_proof(expression, q):
    import sympy as sp
    s=spec_for(q);assert s['kind']=='events'
    p,z=F(s['p']),F(s['p_b']);P,Z=sp.symbols('P Z')
    if p==z:Z=P
    elif p+z==1:Z=1-P
    denominators={p.denominator,z.denominator,100}
    denominators.update(int(d) for _,d in re.findall(r'(\d+)\s*/\s*(\d+)',q['question']))
    aliases={}
    for value,symbol in [(p,P),(1-p,1-P),(z,Z),(1-z,1-Z)]:
        for den in denominators:
            num=value*den
            if num.denominator!=1:continue
            num=int(num)
            for form in (f'{num}/{den}',f'({den}-{den-num})/{den}'):
                key=ast.dump(ast.parse(form,mode='eval').body)
                if key in aliases:assert sp.simplify(aliases[key]-symbol)==0
                aliases[key]=symbol
    def lift(node):
        key=ast.dump(node)
        if key in aliases:return aliases[key]
        if isinstance(node,ast.Constant) and type(node.value) is int and node.value in (0,1,2):return sp.Integer(node.value)
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
            return (-1 if isinstance(node.op,ast.USub) else 1)*lift(node.operand)
        if isinstance(node,ast.BinOp):
            # Exact a/d*b/e reassociation, allowing each supplied ratio to be
            # bound independently; never substitute an answer-only value.
            if isinstance(node.op,ast.Div) and isinstance(node.left,ast.BinOp) and isinstance(node.left.op,ast.Mult):
                ratio=ast.BinOp(left=node.left.right,op=ast.Div(),right=node.right)
                if ast.dump(ratio) in aliases:return lift(node.left.left)*lift(ratio)
            a,b=lift(node.left),lift(node.right)
            if isinstance(node.op,ast.Add):return a+b
            if isinstance(node.op,ast.Sub):return a-b
            if isinstance(node.op,ast.Mult):return a*b
            if isinstance(node.op,ast.Div):return a/b
            if isinstance(node.op,ast.Pow) and b in (0,1,2):return a**b
        raise ValueError('Unbound literal or operation')
    lifted=lift(ast.parse(expression,mode='eval').body)
    expected={'both':P*Z,'neither':(1-P)*(1-Z),'exactly_one':P*(1-Z)+(1-P)*Z,
        'at_least_one':1-(1-P)*(1-Z),'same':P*Z+(1-P)*(1-Z)}[s['target']]
    assert lifted.free_symbols and sp.simplify(lifted-expected)==0
    return dict(method='symbolic_identity_after_explicit_probability_binding',
                lifted=str(lifted),expected=str(expected))


def prove(row,q):
    assert row['review_required'] and row['executable']
    assert F(row['computed'])==oracle(spec_for(q))
    expression=row['normalized_expression']
    if spec_for(q)['kind']=='events':return event_proof(expression,q)
    for reference in partial_refs(q):
        if reviewed_shape(expression)==reviewed_shape(reference):
            return dict(method='affine_second_moment_identity_with_documented_partial_folding',reference=reference)
    raise ValueError('Needs individual semantic review')


def summarize(root):
    repo=Path(__file__).resolve().parents[1]
    data=json.loads((repo/'docs/STATS_SEMANTIC_COURSE_DATA.json').read_text())
    suites=dict(validation_suites(),fresh=data['holdout'])
    manual_path=root/'manual_reviews.json'
    manual=json.loads(manual_path.read_text()) if manual_path.exists() else []
    manual={(r['question_sha256'],r['raw_sha256']):r for r in manual}
    summary, unresolved, proofs = {}, [], []
    for folder in sorted((root/'evaluation').iterdir()):
        metrics={}
        for suite,questions in suites.items():
            path=folder/(suite+'_automatic.json')
            if not path.exists():continue
            rows=json.loads(path.read_text());lookup={q['id']:q for q in questions}
            assert len(rows)==len(questions)
            for row in rows:
                q=lookup[row['id']];assert row['question_sha256']==digest(q)
                if not row.get('review_required'):continue
                key=(row['question_sha256'],row['raw_sha256'])
                if key in manual:
                    r=manual[key];correct=r['correct'];proof=r['reason']
                else:
                    try:proof=prove(row,q);correct=True
                    except (AssertionError,ValueError,KeyError,TypeError,SyntaxError):
                        unresolved.append(dict(model=folder.name,suite=suite,question=q,row=row));continue
                row.update(math_correct=correct,primary_correct=correct,correct=correct,
                           review_required=False,reason='Documented semantic review: '+str(proof))
                proofs.append(dict(model=folder.name,suite=suite,id=row['id'],
                    question_sha256=row['question_sha256'],raw_sha256=row['raw_sha256'],proof=proof,correct=correct))
            (folder/(suite+'_reviewed.json')).write_text(json.dumps(rows,indent=2)+'\n')
            metrics[suite]=suite_metrics(rows)
            if suite=='fresh':
                metrics['fresh_views']={view:suite_metrics([r for r in rows if lookup[r['id']]['view']==view])
                                       for view in ('familiar','verbal')}
        summary[folder.name]=metrics
    gate=json.loads((repo/'docs/STATS_CONSOLIDATION_DECISION_DRAFT.json').read_text())['validation_gate']
    if all(k in summary.get('v15',{}) for k in ('old','chain','event')):
        for arm in ('repeat_control','semantic_course'):
            if all(k in summary.get(arm,{}) for k in ('old','chain','event')):
                hist={k:summary[arm][k] for k in ('old','chain','event')}
                summary[arm]['historical_gate']=gate_result(hist,summary['v15'],gate)
    result=dict(models=summary,pending=len(unresolved),reviews=len(proofs))
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (root/'pending_reviews.json').write_text(json.dumps(unresolved,indent=2)+'\n')
    (root/'semantic_reviews.json').write_text(json.dumps(proofs,indent=2)+'\n')
    print(json.dumps(dict(pending=len(unresolved),reviews=len(proofs),scores={m:{s:x[s]['correct'] for s in ('old','chain','event','fresh') if s in x} for m,x in summary.items()})))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    summarize(p.parse_args().root)
