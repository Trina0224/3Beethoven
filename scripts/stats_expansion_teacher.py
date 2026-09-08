"""Bounded new-curriculum teacher calls with audited intermediates."""
import argparse,hashlib,json,math,re
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
from stats_teacher_ab import MODEL
from stats_teacher_envelope import parse_answers,_unique_object
from stats_consolidation_grader import score,outcome,reviewed_shape,grader_fingerprint
from stats_consolidation_semantics import spec_for,validate_question
from stats_consolidation_pilot import digest
from exact_calculator import calculate
from flight_run_stats_v0_3 import read_json,save_json,read_jsonl,append
from stats_expansion_data import ROOT,corpus

def intermediates(q):
    s=spec_for(q)
    if s['kind']=='events':return {'p_detect_a':F(s['p']),'p_detect_b':F(s['p_b'])}
    return {'mean_Y':F(s['scale'])*F(s['mean'])+F(s['offset']),
            'variance_Y':F(s['scale'])**2*F(s['variance'])}
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
    if len(starts)>=256 or sum(costs)+0.01>0.50:
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


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--run',action='store_true');a=parser.parse_args()
    questions=read_json(ROOT/'docs/STATS_EXPANSION_QUESTIONS.json')['train']
    assert len(questions)==128
    for q in questions:
        validate_question(q)
        fixture={'intermediates':{k:str(v) for k,v in intermediates(q).items()},'answers':[{'question_id':q['id'],'expression':q['expression']}]}
        assert judge(json.dumps(fixture),q)['accepted'],q['id']
        assert q['expression'] not in messages(q)[1]['content']
    if not a.run:print('EXPANSION_TEACHER_FIXTURES_OK');return
    a.root.mkdir(parents=True,exist_ok=True)
    manifest=dict(questions_sha256=digest(questions),protocol_sha256=digest(read_json(ROOT/'docs/STATS_EXPANSION_PROTOCOL.json')),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),grader_fingerprint=grader_fingerprint())
    old=read_json(a.root/'manifest.json');assert old is None or old==manifest
    save_json(a.root/'manifest.json',manifest)
    from kaggle_secrets import UserSecretsClient
    key=UserSecretsClient().get_secret('OPENROUTER_API_KEY');rows=[];attempts=[]
    for q in questions:
        for attempt in (0,1):
            msgs=messages(q)
            if attempt:
                msgs[0]['content']+=' Generic rules: independent events multiply joint probabilities; complement a miss probability to get detection probability. Neither=(1-p)*(1-q), both=p*q, exactly one=p*(1-q)+(1-p)*q, at least one=1-(1-p)*(1-q), same=p*q+(1-p)*(1-q). For Y=a*X+b: E[Y]=a*E[X]+b, Var(Y)=a**2*Var(X), E[Y**2]=Var(Y)+E[Y]**2. Substitute the problem numbers yourself.'
            record=call(a.root,key,q['id']+'_'+str(attempt),msgs)
            assert record['model']==MODEL and record['provider']=='DeepInfra'
            cost=record['usage'].get('cost');assert type(cost) in (int,float) and math.isfinite(cost) and cost>=0
            verdict=judge(record['text'],q,record['finish_reason']) if isinstance(record['text'],str) else dict(accepted=False,classification='non_text')
            row=dict(id=q['id'],target=q['semantics']['target'],domain=q['semantics']['kind'],attempt=attempt,raw=record['text'],**verdict)
            attempts.append(row);save_json(a.root/'attempts.json',attempts)
            if verdict['accepted']:break
        rows.append(row)
        counts=Counter(r['target'] for r in rows if r['accepted'])
        passed=len(rows)==128 and all(counts[t]>=12 for t in ('neither','both','exactly_one','at_least_one','same','mean','variance','second_moment'))
        save_json(a.root/'results.json',dict(completed=len(rows)==128,passed=passed,accepted=sum(counts.values()),by_target=dict(counts),rows=rows))
        ledger=read_jsonl(a.root/'api_ledger.jsonl')
        usage=dict(calls=sum(r['event']=='started' for r in ledger),cost=sum(r['usage']['cost'] for r in ledger if r['event']=='response'))
        save_json(a.root/'usage.json',usage)
        print('EXPANSION_TEACHER',len(rows),q['id'],verdict['classification'],json.dumps(usage),flush=True)
    assert passed,'Insufficient verified targets; no training'
    corpus(a.root)
    print('EXPANSION_TEACHER_COMPLETE',json.dumps(dict(counts)),flush=True)
if __name__=='__main__':main()
