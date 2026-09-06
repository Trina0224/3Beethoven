"""Frozen same-story contrasts with fading symbolic support."""
import json
import random
from pathlib import Path
from stats_curriculum_v0_13 import KINDS,make,digest
from stats_curriculum_v0_14 import reword
from stats_curriculum_v0_15 import score
from exact_calculator import calculate

RULES={
 'mean':'For Y=a*X+b, E[Y]=a*E[X]+b. The offset is added once.',
 'variance':'For Y=a*X+b, Var(Y)=a**2*Var(X). A constant offset contributes zero variance. For Poisson X, Var(X)=E[X]. Do not add b or b**2.',
 'moment':'For Y=a*X+b, E[Y**2]=a**2*Var(X)+(a*E[X]+b)**2. Include both variance and squared mean. For Poisson X, Var(X)=E[X].',
 'scale':'Identify a in Y=a*X+b directly from the story. Do not solve for X or infer a from the mean.'}
CUES={'mean':'Requested quantity: transformed mean.',
 'variance':'Requested quantity: transformed variance, not transformed mean or second moment.',
 'moment':'Requested quantity: second moment, not squared mean.',
 'scale':'Requested quantity: the stated multiplier, not the value of X.'}

def prompt(q,support='none'):
    text=q['question']
    if support!='none':text+='\n'+(RULES if support=='full' else CUES)[q.get('task', 'moment' if q['category']=='moment' else 'variance')]
    return text+'\nReturn only Expression: followed by one fully substituted numerical expression. Keep arithmetic unevaluated. Do not leave X or other variables. No explanation.'

def build():
    docs=Path(__file__).resolve().parents[1]/'docs'
    blocked=set()
    for version in (13,14,15):
        old=json.loads((docs/f'STATS_V0_{version}_FROZEN_QUESTIONS.json').read_text())
        blocked.update(tuple(q['identity']) for rows in old.values() for q in rows)
    for name in ('TEACHER_PERTURBATIONS','TEACHER_SCAFFOLD'):
        old=json.loads((docs/f'STATS_V0_14_{name}.json').read_text())
        blocked.update(tuple(q['identity']) for pair in old['pairs'] for q in pair['questions'])
    rng=random.Random(1616);data={s:[] for s in ('train','validation','test')}
    def fresh(kind,split,i):
        for _ in range(100000):
            q=make(kind,[rng.randrange(12,69),rng.randrange(61,601),rng.randrange(2,8),rng.randrange(2,24)],split,i)
            if tuple(q['identity']) in blocked:continue
            blocked.add(tuple(q['identity']));q['id']=q['id'].replace('v13_','v16_');return q
        raise RuntimeError('No fresh identity')
    for split,n in (('train',24),('validation',6)):
        for family in ('moment','poisson_scaled'):
            for i in range(n):
                q=fresh(family,split,i);b=q['bindings'];m,a,z=(b[k] for k in ('mean','scale','offset'));v=b.get('variance',m)
                if family=='moment':start=f'X has mean {m} and variance {v}. '
                else:start=f'X is a Poisson count with expectation {m}. '
                stories=[start+f'Y={a}*X+{z}. ',start+f'A device multiplies X by {a}, then adds {z} to obtain Y. ',
                    f'Adding {z} after multiplying X by {a} gives Y. '+start]
                story=stories[i%3]
                for task,request,expr in (
                    ('mean','Find the expected value of Y.',f'{a}*{m}+{z}'),
                    ('variance','Find the variance of Y.',f'{a}**2*{v}'),
                    ('moment','Find the expectation of the square of Y.',f'{a}**2*{v}+({a}*{m}+{z})**2'),
                    ('scale','Which number multiplies X to produce Y? Return that multiplier.',a)):
                    if split=='validation' and task=='scale':continue
                    sub=dict(q,id=q['id']+'_'+task,story_id=q['id'],family=family,task=task,
                        category='moment' if task=='moment' else 'affine_'+task,
                        bindings=dict(b,variance=v),question=story+request,expression=expr,answer=calculate(expr))
                    sub.pop('target',None);data[split].append(sub)
    for kind in KINDS:
        for i in range(12):
            q=fresh(kind,'test',i);q['question']=reword(q,True);q.pop('target',None);data['test'].append(q)
    return data

if __name__=='__main__':
    d=build();p=Path(__file__).resolve().parents[1]/'docs/STATS_V0_16_FROZEN_QUESTIONS.json'
    p.write_text(json.dumps(d,indent=2)+'\n');print(digest(d),{s:len(v) for s,v in d.items()})
