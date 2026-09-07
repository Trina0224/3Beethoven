"""First student batch: independently checked teacher outputs, resumable ledger."""
import argparse, json, math, re
from pathlib import Path
from stats_consolidation_pilot import build, digest, TEACHER_MODEL as MODEL, assert_execution_released
from stats_consolidation_grader import score, grader_fingerprint
from formulation_grader import parse_expression
from stats_teacher_envelope import parse_answers
from flight_run_stats_v0_3 import read_json, save_json, read_jsonl, append, package
import prepare_stats_consolidation_teacher as prep
ROOT = Path('/kaggle/working/3beethoven_first_batch_teacher')
PRIOR_CALLS, PRIOR_COST = 110, 0.00806302
def call(root,key,tag,msgs):
    request = {'model':MODEL,'messages':msgs,'temperature':0,'max_tokens':400,
               'provider':{'only':['deepinfra'],'allow_fallbacks':False,
               'require_parameters':True,'enforce_distillable_text':True,'max_price':{'prompt':1,'completion':2,'request':0}}}
    if 'No JSON.' not in msgs[0]['content']:
        request['response_format']={'type':'json_object'}
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
    if len(starts)>=192 or PRIOR_CALLS+len(starts)>=302 or PRIOR_COST+sum(costs)+0.01>3.0:
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


def parse(raw, ids):
    text = raw.strip()
    if text.startswith('```json\n') and text.endswith('\n```') and text.count('```') == 2:
        text = text[8:-4].strip()
    return parse_answers(text, ids)


def messages(request, pending, retry=False):
    system = ('Return only compact JSON: {"answers":[{"question_id":"id","expression":"numerical expression"}]}. '
              'No prose, no derivation, no echoed question or schema. Solve every requested question. '
              'Use numbers,+,-,*,/,**,comb(n,r). Keep arithmetic, unit conversions, complements and powers unevaluated. '
              'Do not use variables. General checks: a Poisson process count mean is rate times duration in matching units; '
              'its variance equals that mean. E[Z^2]=Var(Z)+E[Z]^2. Var(aX+b)=a^2 Var(X). '
              'A confidence interval keeps its center when sample size changes; multiplying sample size by k divides '
              'the old half-width by sqrt(k), never multiplies it. Exactly one of independent A,B is '
              'P(A)(1-P(B))+(1-P(A))P(B). Conditional total waiting time includes time already elapsed. '
              'These are general rules; substitute the numbers yourself.')
    if retry:
        system += ' Previous output was rejected. Recheck units, requested quantity, and every expression; supply all requested IDs.'
    if request['archetype']=='interval':
        system = ('Return only compact JSON with center, old_half_width, new_half_width strings and an answers array '
                  'containing question_id and expression strings. Use these intermediate fields as your entire working. '
                  'The center is the average of the two old endpoints. The old half-width is half their difference. '
                  'The new half-width is the old half-width divided by the square root of the sample-size multiplier. '
                  'The new upper endpoint is center plus NEW half-width. Use sqrt() for square roots. '
                  'Substitute all numbers; no variables, prose, explanation or question echo. Keep the whole response under 180 tokens.')
    elif request['archetype']=='poisson_process':
        system += (' For rate r per minute and duration t seconds, mean m=r*(t/60). '
                   'Count variance=m, count second moment=m+m**2. Substitute the FULL numerical m in both places; '
                   'do not output Var(X), m or any symbols. Convert either rate or duration, never both.')
    user = {'questions':[q for q in request['questions'] if q['question_id'] in pending]}
    return [{'role':'system','content':system},{'role':'user','content':json.dumps(user,separators=(',',':'))}]


def ingest(record, raw, qlookup, attempt, finish='stop'):
    a = dict(attempt=attempt, raw=raw, finish_reason=finish, parsed={}, judgements={})
    try:
        if finish != 'stop':
            raise ValueError('Truncated or non-stop output')
        if len(qlookup)==1 and re.fullmatch(r'Expression:\s*[^\n]+',raw.strip()):
            items=[{'question_id':next(iter(qlookup)), 'expression':raw.strip().split(':',1)[1].strip()}]
            repair='single_question_expression_line_transport'
        elif len(qlookup)==1 and '\n' not in raw.strip() and not raw.strip().startswith('{'):
            parse_expression(raw.strip(), {})
            items=[{'question_id':next(iter(qlookup)), 'expression':raw.strip()}]
            repair='single_question_bare_expression_transport'
        else:
            items, repair = parse(raw, set(qlookup))
        a['transport_repair'] = repair
        for item in items:
            qid, expression = item['question_id'], item['expression'].strip()
            a['parsed'][qid] = expression
            judged = score('Expression: '+expression, qlookup[qid])
            a['judgements'][qid] = judged
            if judged['primary_correct'] and qid not in record['accepted']:
                record['accepted'][qid] = dict(teacher_expression=expression,
                    normalized_expression=judged['normalized_expression'], judged=judged, attempt=attempt)
    except (ValueError, KeyError, TypeError, SyntaxError) as exc:
        a['error'] = str(exc)
    record['attempts'].append(a)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--source',type=Path,default=Path('/kaggle/working/3beethoven_stats_consolidation_teacher'))
    args = parser.parse_args()
    config = read_json(prep.DECISION)
    assert_execution_released(config)
    stories, requests, _, _ = build()
    if not args.run:
        for r in requests:
            assert sum(len(m['content'].encode()) for m in messages(r,{q['question_id'] for q in r['questions']})) <= 4000
        print('FIRST_BATCH_VALIDATED',len(stories)); return
    ROOT.mkdir(parents=True,exist_ok=True)
    prep.ROOT = ROOT
    contract = dict(decision_sha256=digest(config),request_sha256=digest(requests),grader_fingerprint=grader_fingerprint(),
                    prior_calls=PRIOR_CALLS,prior_cost=PRIOR_COST,rule_assisted=True,reference_conditioned=False)
    old = read_json(ROOT/'contract.json')
    if old and old != contract:
        raise RuntimeError('Contract drift')
    save_json(ROOT/'contract.json',contract)
    lookup = {s['story_id']:s for s in stories}
    from kaggle_secrets import UserSecretsClient
    key = UserSecretsClient().get_secret('OPENROUTER_API_KEY')
    try:
        for scope in ('pilot','full'):
            for r in requests:
                if scope=='pilot' and not r['pilot_batch']:
                    continue
                path = ROOT/'records'/(r['story_id']+'.json')
                record = read_json(path)
                qs = {q['id']:q for q in lookup[r['story_id']]['questions']}
                if record is None:
                    record = dict(story_id=r['story_id'],request_sha256=digest(r),attempts=[],accepted={})
                    legacy = read_json(args.source/'records'/(r['story_id']+'.json'))
                    if legacy and legacy.get('request_sha256')==digest(r):
                        for i,a in enumerate(legacy['attempts']):
                            ingest(record,a['raw'],qs,'legacy_'+str(i))
                        record['legacy_source']=str(args.source)
                    save_json(path,record)
                if record['request_sha256'] != digest(r):
                    raise RuntimeError('Request drift')
                new_attempts = sum(isinstance(a['attempt'],int) for a in record['attempts'])
                # Reparse existing original strings after a transport-only extension.
                if r['archetype']=='interval':
                    snapshot=list(record['attempts'])
                    for a in snapshot:
                        if a.get('attempt')==3 and not a.get('parsed'):
                            ingest(record,a['raw'],qs,'transport_review_3',a['finish_reason'])
                    save_json(path,record)
                for attempt in range(new_attempts,5 if r['archetype']=='interval' else 3):
                    pending = set(qs)-set(record['accepted'])
                    if not pending:
                        break
                    result = call(ROOT,key,r['story_id']+'_'+str(attempt),messages(r,pending,attempt>0))
                    if result['model']!=MODEL or result['provider']!='DeepInfra':
                        raise RuntimeError('Teacher identity drift')
                    ingest(record,result['text'],qs,attempt,result['finish_reason'])
                    save_json(path,record)
                print('FIRST_BATCH_TEACHER',scope,r['story_id'],len(record['accepted']),'/',len(qs),flush=True)
            verified = prep.build_verified(stories)
            for rows in verified.values():
                for row in rows:
                    rec=read_json(ROOT/'records'/(row['story_id']+'.json'))
                    row['rule_assisted']=isinstance(rec['accepted'][row['source_id']]['attempt'],int)
            save_json(ROOT/'verified_teacher_rows.json',verified)
            report = prep.acceptance_report(stories,scope)
            gate = prep.gate_acceptance(report,config['teacher'])
            gate.update(contract,scope=scope,verified_rows_sha256=digest(verified))
            save_json(ROOT/(scope+'_acceptance.json'),report)
            save_json(ROOT/(scope+'_gate.json'),gate)
            print('FIRST_BATCH_GATE',scope,json.dumps(gate),flush=True)
            if not gate['passed']:
                raise RuntimeError(scope+' gate failed; inspect preserved rejects before further calls')
    finally:
        ledger=read_jsonl(ROOT/'api_ledger.jsonl')
        save_json(ROOT/'usage.json',dict(prior_calls=PRIOR_CALLS,prior_cost=PRIOR_COST,
             new_calls=sum(r['event']=='started' for r in ledger),
             new_cost=sum(r['usage'].get('cost',0) or 0 for r in ledger if r['event']=='response')))
        package(ROOT)


if __name__=='__main__':
    main()
