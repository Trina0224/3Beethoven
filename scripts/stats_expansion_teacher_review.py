"""Conservative raw-preserving envelope recovery and symbolic event review.

A post-output engineering review, not a change to the student grader or gate.
"""
import ast,json,hashlib,argparse
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
from stats_teacher_envelope import _unique_object,parse_answers
from stats_expansion_teacher_format import intermediates
from stats_consolidation_grader import score
from stats_consolidation_semantics import spec_for
from stats_consolidation_pilot import digest
from exact_calculator import calculate

def envelope(raw,q):
    decoder=json.JSONDecoder(object_pairs_hook=_unique_object)
    obj,end=decoder.raw_decode(raw.strip());tail=raw.strip()[end:].strip();repairs=[]
    if tail:
        assert tail=='}','Unrecognized trailing text';repairs.append('one_redundant_trailing_closing_brace')
    if set(obj)=={'question_id','question','output_structure'}:
        assert obj['question_id']==q['id'] and obj['question']==q['question'],'Echo mismatch'
        obj=obj['output_structure'];repairs.append('unique_echoed_output_structure')
    assert set(obj)=={'intermediates','answers'}
    return obj,repairs

def event_proof(expression,q):
    import sympy as sp
    s=spec_for(q);assert s['kind']=='events'
    p,z=F(s['p']),F(s['p_b']);P,Z=sp.symbols('p_a p_b')
    if p==z:Z=P
    elif p+z==1:Z=1-P
    aliases={}
    for value,symbol in [(p,P),(1-p,1-P),(z,Z),(1-z,1-Z)]:
        num=value.numerator;den=value.denominator;percent=value*100;assert percent.denominator==1
        percent=int(percent)
        forms=[f'{num}/{den}',f'{percent}/100',f'(100-{100-percent})/100']
        for form in forms:
            key=ast.dump(ast.parse(form,mode='eval').body)
            if key in aliases:assert sp.simplify(aliases[key]-symbol)==0
            aliases[key]=symbol
    def lift(node):
        key=ast.dump(node)
        if key in aliases:return aliases[key]
        if isinstance(node,ast.Constant) and type(node.value) is int and node.value in (0,1,2):return sp.Integer(node.value)
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            return lift(node.operand)*(1 if isinstance(node.op,ast.UAdd) else -1)
        if isinstance(node,ast.BinOp):
            a,b=lift(node.left),lift(node.right)
            if isinstance(node.op,ast.Add):return a+b
            if isinstance(node.op,ast.Sub):return a-b
            if isinstance(node.op,ast.Mult):return a*b
            if isinstance(node.op,ast.Div):return a/b
            if isinstance(node.op,ast.Pow) and b in (0,1,2):return a**b
        raise ValueError('Expression contains a literal or operation without an input-binding proof')
    lifted=lift(ast.parse(expression,mode='eval').body)
    expected={'both':P*Z,'neither':(1-P)*(1-Z),'exactly_one':P*(1-Z)+(1-P)*Z,'at_least_one':1-(1-P)*(1-Z),'same':P*Z+(1-P)*(1-Z)}[s['target']]
    assert lifted.free_symbols and sp.simplify(lifted-expected)==0,'Event identity differs'
    return {'method':'symbolic_event_identity_after_explicit_percentage_binding','lifted':str(lifted),'expected':str(expected),'assumption':'independent detections; complement aliases derive from stated inputs'}

def review(record,q):
    result={'accepted':False,'classification':'review_rejected'}
    try:
        assert record['finish_reason']=='stop' and isinstance(record['text'],str)
        obj,repairs=envelope(record['text'],q)
        items,_=parse_answers(json.dumps({'answers':obj['answers']}),{q['id']});assert len(items)==1
        expected=intermediates(q);steps=obj['intermediates'];excluded={}
        if spec_for(q)['kind']=='moments' and spec_for(q)['target']=='variance':
            assert set(steps) in ({'variance_Y'},{'mean_Y','variance_Y'})
            if 'mean_Y' in steps:excluded={'mean_Y':{'raw':steps['mean_Y'],'reason':'E[X] not supplied; unsupported working is excluded, not endorsed'}}
            expected={'variance_Y':expected['variance_Y']}
        else:assert set(steps)==set(expected)
        checks={k:isinstance(steps[k],str) and F(calculate(steps[k]))==v for k,v in expected.items()};assert all(checks.values())
        expression=items[0]['expression'];judged=score('Expression: '+expression,q)
        assert judged['executable'] and F(judged['computed'])==F(q['answer'])
        proof={'method':'unchanged_primary_grader'}
        if not judged['primary_correct']:proof=event_proof(expression,q)
        return dict(accepted=True,classification='accepted_after_documented_review',expression=expression,intermediate_checks=checks,
                    repairs=repairs,excluded_unsupported_working=excluded,full_working_endorsed=not bool(excluded),proof=proof,question_sha256=digest(q),raw_sha256=hashlib.sha256(record['text'].encode()).hexdigest(),judged=judged)
    except (ValueError,AssertionError,KeyError,TypeError,SyntaxError) as exc:
        return dict(result,error=str(exc))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];questions=json.loads((repo/'docs/STATS_EXPANSION_QUESTIONS.json').read_text())['train']
    resultpath=a.root/'results.json';original=a.root/'results_before_semantic_review.json'
    if not original.exists():original.write_bytes(resultpath.read_bytes())
    rows=[];allreviews=[]
    for q in questions:
        options=[]
        for attempt in (0,1):
            path=a.root/'api_cache'/(q['id']+'_'+str(attempt)+'.json')
            if not path.exists():continue
            record=json.loads(path.read_text());r=dict(id=q['id'],target=q['semantics']['target'],attempt=attempt,raw=record['text'],**review(record,q));allreviews.append(r);options.append(r)
        if options:rows.append(next((r for r in options if r['accepted']),options[-1]))
        else:rows.append(dict(id=q['id'],target=q['semantics']['target'],accepted=False,classification='no_cached_response_service_limited_collection'))
    counts=Counter(r['target'] for r in rows if r['accepted']);passed=all(counts[t]>=12 for t in ('neither','both','exactly_one','at_least_one','same','mean','variance','second_moment'))
    (a.root/'semantic_reviews.json').write_text(json.dumps(allreviews,indent=2)+'\n')
    resultpath.write_text(json.dumps(dict(completed=True,teacher_collection_complete=False,collection_stop='STATS_EXPANSION_COLLECTION_STOP.json',cached_question_count=sum(any(r['id']==q['id'] for r in allreviews) for q in questions),passed=passed,accepted=sum(counts.values()),by_target=dict(counts),review_type='post-output raw-preserving engineering review; unchanged student grader',rows=rows),indent=2)+'\n')
    print('EXPANSION_TEACHER_REVIEW',json.dumps({'passed':passed,'by_target':dict(counts),'accepted':sum(counts.values())}),flush=True)
    assert passed,'Minimum teacher coverage still not met'
    from stats_expansion_data import corpus
    corpus(a.root)
if __name__=='__main__':main()
