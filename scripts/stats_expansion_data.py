"""Preregistered new task identities; references are checks, never teacher targets."""
import json, random, math
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
from stats_consolidation_semantics import task_key,oracle,validate_question
from stats_consolidation_pilot import digest
from stats_multiview_corpus import render, SUFFIX
from stats_consolidation_grader import score
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2)+'\n')
def walk(x):
    if isinstance(x,dict):
        if all(k in x for k in ('question','expression','answer','category','id')):yield x
        for v in x.values():yield from walk(v)
    elif isinstance(x,list):
        for v in x:yield from walk(v)
def make(domain,params,target,split,i):
    sid=f'expansion_{split}_{domain}_{i:02}'
    if domain=='events':
        a,b,miss=params;p,z=f'{a}/100',f'{b}/100'
        s=dict(kind='events',target=target,p=p,p_b=z)
        bindings={'miss_a':str(1-F(p)),'miss_b':str(1-F(z))} if miss else {}
        if miss:
            pa,pb=f'(1-{100-a}/100)',f'(1-{100-b}/100)'
        else:pa,pb=f'({p})',f'({z})'
        ex={'neither':f'(1-{pa})*(1-{pb})','both':f'{pa}*{pb}',
            'exactly_one':f'{pa}*(1-{pb})+(1-{pa})*{pb}',
            'at_least_one':f'1-(1-{pa})*(1-{pb})','same':f'{pa}*{pb}+(1-{pa})*(1-{pb})'}[target]
        cat=target
    else:
        m,v,a,b=params;s=dict(kind='moments',target=target,mean=str(m),variance=str(v),scale=str(a),offset=str(b));bindings={}
        ex={'mean':f'({a})*({m})+({b})','variance':f'({a})**2*({v})',
            'second_moment':f'({a})**2*({v})+(({a})*({m})+({b}))**2'}[target]
        cat='moment_'+target
    q=dict(id=sid+'_'+target,story_id=sid,category=cat,semantics=s,bindings=bindings,expression=ex,answer=str(oracle(s)),identity=[sid,target],parameters=list(params))
    q['question']=render(q,0 if split=='train' else 1)
    validate_question(q);assert score('Expression: '+ex,q)['primary_correct'],q
    return q

def generate():
    protocol=read(ROOT/'docs/STATS_EXPANSION_PROTOCOL.json');assert protocol['status']=='frozen_before_data_generation'
    blocked=set();files=[]
    for path in sorted((ROOT/'docs').glob('*.json')):
        if path.name.startswith('STATS_EXPANSION'):continue
        try:data=read(path)
        except (ValueError,UnicodeError):continue
        count=0
        for q in walk(data):
            try:blocked.add(task_key(q));count+=1
            except (ValueError,KeyError,TypeError,ZeroDivisionError):continue
        if count:files.append({'file':path.name,'sha256':digest(data),'questions_scanned':count})
    prior=read(ROOT/'docs/STATS_MULTIVIEW_CORPUS.json')
    # JSON-loaded task keys contain nested arrays: normalize back to tuples.
    def frozen(x):return tuple(frozen(v) for v in x) if isinstance(x,list) else x
    blocked.update(frozen(r['semantic_key']) for r in prior['train'])
    rng=random.Random(560056);result={'train':[],'transfer_probe':[]};initial=len(blocked)
    for split,n in [('train',16),('diagnostic',8)]:
        for domain,targets in [('events',protocol['teacher']['event_targets']),('moments',protocol['teacher']['moment_targets'])]:
            for i in range(n):
                for attempt in range(100000):
                    params=(rng.randrange(7,94),rng.randrange(7,94),bool(i%2)) if domain=='events' else (rng.randrange(11,98),rng.randrange(13,299),rng.randrange(2,10),rng.randrange(2,31))
                    qs=[make(domain,params,t,split,i) for t in targets]
                    keys={task_key(q) for q in qs}
                    if len(keys)==len(qs) and not keys&blocked:break
                else:raise RuntimeError('Identity exhaustion')
                blocked.update(keys);result['train' if split=='train' else 'transfer_probe'].extend(qs)
    assert len(result['train'])==128 and len(result['transfer_probe'])==64
    result['manifest']=dict(protocol_sha256=digest(protocol),initial_blocked_identities=initial,source_files=files,
        train_sha256=digest(result['train']),probe_sha256=digest(result['transfer_probe']),new_identities=192,overlap=0)
    save(ROOT/'docs/STATS_EXPANSION_QUESTIONS.json',result)
    print(json.dumps({k:v for k,v in result['manifest'].items() if k!='source_files'}))

def corpus(teacher_root):
    data=read(ROOT/'docs/STATS_EXPANSION_QUESTIONS.json');results=read(teacher_root/'results.json')
    assert results['completed'] and results['passed']
    accepted={r['id']:r for r in results['rows'] if r['accepted']};prior=read(ROOT/'docs/STATS_MULTIVIEW_CORPUS.json');rows=list(prior['train'])
    for q in data['train']:
        if q['id'] not in accepted:continue
        r=accepted[q['id']];target='Expression: '+r['expression']
        if not score(target,q)['primary_correct']:
            from stats_expansion_teacher_review import event_proof
            assert r['accepted'] and r['question_sha256']==digest(q) and event_proof(r['expression'],q)==r['proof']
        for view in (0,1):
            rows.append(dict(source_id=q['id'],story_id=q['story_id'],category=q['category'],source_group='expansion',view=f'new_{view}',
                prompt=render(q,view)+SUFFIX,target=target,semantic_key=task_key(q),source_row_sha256=digest(r),original_question_sha256=digest(q),
                target_sha256=__import__('hashlib').sha256(target.encode()).hexdigest(),provenance='verified_70B_new_target'))
    assert rows[:516]==prior['train'] and len(rows)<=772
    out=dict(train=rows,transfer_probe=data['transfer_probe'],manifest=dict(accepted_new_targets=len(accepted),train_sha256=digest(rows),probe_sha256=digest(data['transfer_probe']),teacher_results_sha256=digest(results)))
    save(ROOT/'docs/STATS_EXPANSION_CORPUS.json',out)
    protocol=read(ROOT/'docs/STATS_EXPANSION_PROTOCOL.json');steps=math.ceil(len(rows)/8)
    decision=dict(protocol,status='frozen_for_execution',train_sha256=digest(rows),probe_sha256=digest(data['transfer_probe']),validation_steps=sorted(set([12,24,48,steps])),final_step=steps)
    save(ROOT/'docs/STATS_EXPANSION_DECISION.json',decision)
    print('EXPANSION_CORPUS',json.dumps(out['manifest']),flush=True)
if __name__=='__main__':generate()
