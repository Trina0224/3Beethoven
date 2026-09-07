"""Compact-output continuation; old A/B code, raw answers and gates stay frozen."""
import argparse
import hashlib
import json
import math
import re
from collections import Counter
from fractions import Fraction as F
from pathlib import Path

from stats_teacher_ab import MODEL, DATA, DOCS, intermediates
from stats_teacher_envelope import parse_answers, _unique_object
from stats_consolidation_grader import score, outcome, reviewed_shape, grader_fingerprint
from stats_consolidation_semantics import spec_for, validate_question
from stats_consolidation_pilot import digest
from exact_calculator import calculate
from flight_run_stats_v0_3 import read_json, save_json, read_jsonl, append, package

ROOT = Path('/kaggle/working/3beethoven_teacher_compact')
PRIOR_CALLS = 86
PRIOR_COST = 0.00693694


def payload(raw):
    """Extract a unique complete envelope; do not modify any expression bytes."""
    text, repair = raw.strip(), None
    if '```' in text:
        blocks = re.findall(r'```(?:json)?\s*(.*?)\s*```', text, re.S)
        if len(blocks) != 1 or text.count('```') != 2:
            raise ValueError('Ambiguous or incomplete fenced JSON')
        outside = re.sub(r'```(?:json)?\s*.*?\s*```', '', text, flags=re.S)
        if any(marker in outside for marker in ('{', '}', 'Expression:', '"answers"', '"expression"')):
            raise ValueError('Competing answer outside JSON')
        text, repair = blocks[0], 'extracted_unique_fenced_JSON'
    obj = json.loads(text, object_pairs_hook=_unique_object)
    if not isinstance(obj, dict) or set(obj) != {'intermediates', 'answers'}:
        raise ValueError('Unexpected envelope keys')
    return obj, repair


def partial_refs(q):
    s = spec_for(q)
    if s['kind'] not in ('poisson', 'moments') or s.get('target') != 'second_moment' or 'scale' not in s:
        return []
    a,m,b,v = (F(s[k]) for k in ('scale', 'mean', 'offset', 'variance')) if 'variance' in s else (F(s['scale']),F(s['mean']),F(s['offset']),F(s['mean']))
    def n(x):
        return str(x.numerator) if x.denominator == 1 else f'({x.numerator}/{x.denominator})'
    refs = [f'{n(a*a)}*{n(v)}+({n(a)}*{n(m)}+{n(b)})**2']
    for cross in (f'2*{n(a)}*{n(b)}*{n(m)}',f'{n(2*a)}*{n(b)}*{n(m)}',f'{n(2*a*b)}*{n(m)}'):
        for square in (f'{n(b)}**2',n(b*b)):
            refs.append(f'{n(a*a)}*({n(v)}+{n(m)}**2)+{cross}+{square}')
    return refs


def judge(raw, q, finish='stop'):
    result = dict(accepted=False, classification='malformed_response', strict_json=False)
    if finish != 'stop':
        return dict(result, classification='truncated_or_non_stop', finish_reason=finish)
    try:
        obj, repair = payload(raw)
        result.update(strict_json=repair is None, transport_repair=repair)
        items,_ = parse_answers(json.dumps({'answers':obj['answers']}), {q['id']})
        if len(items) != 1:
            raise ValueError('Expected exactly one answer')
        expected = intermediates(q)
        steps = obj['intermediates']
        if not isinstance(steps,dict) or set(steps) != set(expected):
            raise ValueError('Missing or unexpected intermediate keys')
        checks = {k: isinstance(steps[k],str) and F(calculate(steps[k])) == val for k,val in expected.items()}
        expression = items[0]['expression']
        scored = score('Expression: '+expression,q)
        if scored['executable'] and scored['math_correct'] is None and F(scored['computed']) == F(q['answer']):
            if any(reviewed_shape(expression) == reviewed_shape(ref) for ref in partial_refs(q)):
                scored.update(math_correct=True,primary_correct=True,correct=True,review_required=False,
                              reason='Preregistered partially folded affine identity with independent exact agreement')
        result.update(expression=expression, intermediate_checks=checks, judged=scored,
                      classification=outcome(scored), accepted=outcome(scored)=='accepted' and all(checks.values()))
        if not all(checks.values()):
            result['classification']='intermediate_error'
    except (ValueError,KeyError,TypeError,SyntaxError,ZeroDivisionError) as exc:
        result.update(accepted=False,classification='malformed_response',error=str(exc))
    return result


def messages(q):
    structure = {'intermediates':{k:'<numerical expression>' for k in intermediates(q)},
                 'answers':[{'question_id':q['id'],'expression':'<numerical expression>'}]}
    system = ('Return ONLY one compact JSON object. Start with { and end with }. '
              'No introduction, explanation, Markdown, code fence, derivation or text outside JSON. '
              'Fill every requested field with a short numerical expression string. '
              'Keep arithmetic in expressions unevaluated; use only numbers, +,-,*,/,**,comb(n,r). '
              'Replace variables by the supplied numbers. Do not output E[...], Var(...), or placeholders. '
              'The intermediate fields are the entire working: do not explain them. '
              'Keep the whole response under 180 tokens.')
    user = {'question_id':q['id'],'question':q['question'],'output_structure':structure}
    return [{'role':'system','content':system},{'role':'user','content':json.dumps(user,separators=(',',':'))}]


def call(root,key,tag,msgs):
    request = {'model':MODEL,'messages':msgs,'temperature':0,'max_tokens':400,
               'response_format':{'type':'json_object'},'provider':{'only':['deepinfra'],'allow_fallbacks':False,
               'require_parameters':True,'enforce_distillable_text':True,'max_price':{'prompt':1,'completion':2,'request':0}}}
    signature = digest(request)
    cache,ledger = root/'api_cache'/(tag+'.json'),root/'api_ledger.jsonl'
    prior = read_json(cache)
    if prior:
        if prior['request_sha256'] != signature:
            raise RuntimeError('Changed request for cached response')
        return prior
    history = read_jsonl(ledger)
    starts = [r['tag'] for r in history if r['event']=='started']
    done = [r for r in history if r['event']=='response']
    if set(starts)-{r['tag'] for r in done}:
        raise RuntimeError('Unresolved attempt; refusing duplicate billing')
    costs = [r['usage'].get('cost') for r in done]
    if any(type(c) not in (int,float) or not math.isfinite(c) or c<0 for c in costs):
        raise RuntimeError('Invalid or missing cost')
    if len(starts)>=24 or PRIOR_CALLS+len(starts)>=216 or PRIOR_COST+sum(costs)+0.01>0.50:
        raise RuntimeError('Spending/call cap reached')
    if sum(len(m['content'].encode()) for m in msgs)>4000:
        raise RuntimeError('Prompt byte cap reached')
    import requests
    append(ledger,dict(event='started',tag=tag,request_sha256=signature))
    try:
        response=requests.post('https://openrouter.ai/api/v1/chat/completions',headers={'Authorization':'Bearer '+key},json=request,timeout=90)
    except requests.RequestException:
        raise RuntimeError('Network failure; no automatic retry') from None
    if response.status_code!=200:
        append(ledger,dict(event='http_error',tag=tag,status=response.status_code))
        raise RuntimeError(f'HTTP {response.status_code}; no automatic retry; response body omitted')
    body=response.json()
    choice=body.get('choices',[{}])[0]
    result=dict(request=request,request_sha256=signature,text=choice.get('message',{}).get('content'),
                finish_reason=choice.get('finish_reason'),usage=body.get('usage',{}),model=body.get('model'),
                provider=body.get('provider'),id=body.get('id'))
    save_json(cache,result)
    append(ledger,dict(event='response',tag=tag,usage=result['usage']))
    return result


def summary(rows,complete):
    counts=Counter(r['family'] for r in rows if r['accepted'])
    pending=[r['id'] for r in rows if r['classification']=='equivalence_pending']
    passed=complete and sum(counts.values())>=22 and all(counts[f]>=7 for f in ('poisson_process','interval','affine_poisson')) and not pending
    return dict(completed=complete,accepted=sum(counts.values()),n=len(rows),by_family=dict(counts),
                strict_json=sum(r['strict_json'] for r in rows),classifications=dict(Counter(r['classification'] for r in rows)),
                pending=pending,passed=passed,role='reused_development_engineering_check',rows=rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    questions=read_json(DATA)['questions']
    evidence=read_json(DOCS/'STATS_TEACHER_AB_EVIDENCE.json')
    assert len(questions)==24 and len(evidence['responses'])==48
    for q in questions:
        validate_question(q)
        fixture={'intermediates':{k:str(v) for k,v in intermediates(q).items()},'answers':[{'question_id':q['id'],'expression':q['expression']}]}
        assert judge(json.dumps(fixture),q)['accepted'],q['id']
    old_rows=[]
    for q in questions:
        record=next(r for r in evidence['responses'] if json.loads(r['request']['messages'][1]['content'])['question_id']==q['id'] and 'intermediates object' in r['request']['messages'][0]['content'])
        old_rows.append(dict(id=q['id'],family=q['development_family'],**judge(record['text'],q,record['finish_reason'])))
    old=summary(old_rows,True)
    print('OLD_B_COMMON_CONTRACT',json.dumps({k:v for k,v in old.items() if k!='rows'}),flush=True)
    if not args.run:
        print('VALIDATED 24 compact fixtures; no calls')
        return
    ROOT.mkdir(exist_ok=True,parents=True)
    manifest=dict(data_sha256=digest(read_json(DATA)),grader_fingerprint=grader_fingerprint(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),prior_calls=PRIOR_CALLS,prior_cost=PRIOR_COST)
    prior=read_json(ROOT/'run_manifest.json')
    if prior and prior!=manifest:
        raise RuntimeError('Run manifest mismatch')
    save_json(ROOT/'run_manifest.json',manifest)
    save_json(ROOT/'old_B_common_contract.json',old)
    from kaggle_secrets import UserSecretsClient
    key=UserSecretsClient().get_secret('OPENROUTER_API_KEY')
    rows,complete=[],False
    try:
        for q in questions:
            record=call(ROOT,key,q['id']+'_C',messages(q))
            if record['model']!=MODEL or record['provider']!='DeepInfra':
                raise RuntimeError('Provider/model pin mismatch')
            cost=record['usage'].get('cost')
            if type(cost) not in (int,float) or not math.isfinite(cost) or cost<0:
                raise RuntimeError('Invalid or missing cost')
            verdict=judge(record['text'],q,record['finish_reason']) if isinstance(record['text'],str) else dict(accepted=False,strict_json=False,classification='non_text')
            rows.append(dict(id=q['id'],family=q['development_family'],raw=record['text'],**verdict))
            save_json(ROOT/'results.json',summary(rows,False))
            print('COMPACT',len(rows),'/24',q['id'],verdict['classification'],flush=True)
        complete=True
    except Exception as exc:
        save_json(ROOT/'stop.json',dict(type=type(exc).__name__,error=str(exc)))
        raise
    finally:
        result=summary(rows,complete)
        baseline={r['id']:r for r in old_rows}
        result['paired']=dict(Counter(f"B{int(baseline[r['id']]['accepted'])}_C{int(r['accepted'])}" for r in rows))
        save_json(ROOT/'results.json',result)
        history=read_jsonl(ROOT/'api_ledger.jsonl')
        save_json(ROOT/'usage.json',dict(prior_calls=PRIOR_CALLS,prior_cost=PRIOR_COST,
                  calls=sum(r['event']=='started' for r in history),
                  new_cost=sum(r['usage'].get('cost',0) or 0 for r in history if r['event']=='response')))
        package(ROOT)
        print('COMPACT_SUMMARY',json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    main()
