"""New frozen curriculum: decompose weak concepts; keep test identities separate."""
import json
import random
from pathlib import Path
from stats_curriculum_v0_13 import KINDS,make,digest
from stats_curriculum_v0_14 import reword
from prepare_verified_distillation_v0_14 import formulation_prompt as prompt
from exact_calculator import calculate
from formulation_grader import grade,shape,parse_expression

WEAK=('moment','poisson_scaled','poisson_time','uniform_time')
RULES=('General rules: For a Poisson process the count variance is rate times duration in the same time unit. '
       'Var(a*X+b)=a**2*Var(X); a constant offset does not change variance. '
       'E[a*X+b]=a*E[X]+b. E[(a*X+b)**2]=a**2*Var(X)+(a*E[X]+b)**2. '
       'E[X**2]=Var(X)+E[X]**2; the squared mean must not be omitted. '
       'For T uniform on [0,U], E[T given T>c]=(c+U)/2, in consistent units; this is total wait. '
       'For n independent Bernoulli trials, P(K=r)=comb(n,r)*p**r*(1-p)**(n-r). '
       'For independent miss probabilities p,q, exactly one detection is (1-p)*q+p*(1-q); at least one is 1-p*q. '
       'Increasing sample size by k**2 shrinks a normal interval half-width by k, keeping its center. '
       'These symbolic rules are reminders, not numeric answers. Return only Expression: followed by a fully '
       'substituted numerical expression. Keep arithmetic, complements, powers and conversion factors unevaluated.')


def score(raw,q):
    result=grade(raw.replace('Comb(','comb('),q)
    # Frozen extensions cover equivalences identified in the PREVIOUS run.
    # Exact agreement alone never earns credit.
    if result['math_correct'] is None and result['executable']:
        b=q['bindings'];c=q['category'];refs=[]
        if c=='poisson_time':
            seconds=b['duration_minutes'].split('/')[0]
            refs=[f"{b['rate_per_minute']}*{seconds}/60"]
        elif c=='binomial':
            n,r,p=b['n'],b['r'],b['reject_probability'];complement=calculate('1-('+p+')')
            for pp in (f'1-({p})',complement):
                for exp in (f'{n}-{r}',str(int(n)-int(r))):
                    refs.append(f'comb({n},{r})*({p})**{r}*({pp})**({exp})')
        elif c=='at_least_one':
            p,z=b['miss_a'],b['miss_b']
            refs=[f'(1-{p})*(1-{z})+{p}*(1-{z})+(1-{p})*{z}']
        elif c=='exactly_one':
            p,z=b['miss_a'],b['miss_b'];a=calculate('1-'+p);d=calculate('1-'+z)
            refs=[f'{a}*{z}+{p}*{d}']
        if result.get('computed')==q['answer'] and shape(parse_expression(result['normalized_expression'],{})) in {shape(parse_expression(x,{})) for x in refs}:
            result.update(math_correct=True,review_required=False,reason='Previously reviewed mathematical equivalence; exact value also verified.')
    return dict(result,correct=result['math_correct'] is True,invalid=not result['executable'],grader_version='formulation-v2-frozen-before-v15')


def build():
    docs=Path(__file__).resolve().parents[1]/'docs';blocked=set()
    for v in (13,14):
        data=json.loads((docs/f'STATS_V0_{v}_FROZEN_QUESTIONS.json').read_text())
        blocked.update(tuple(q['identity']) for rows in data.values() for q in rows)
    for name in ('TEACHER_PERTURBATIONS','TEACHER_SCAFFOLD'):
        data=json.loads((docs/f'STATS_V0_14_{name}.json').read_text())
        blocked.update(tuple(q['identity']) for p in data['pairs'] for q in p['questions'])
    rng=random.Random(1515);out={s:[] for s in ('train','validation','test')}
    for split,count in (('train',12),('validation',4),('test',8)):
        for kind in KINDS:
            n=count+(12 if split=='train' and kind in WEAK else 0)
            for i in range(n):
                while True:
                    p=[rng.randrange(12,69),rng.randrange(61,601),rng.randrange(2,8),rng.randrange(2,24)]
                    q=make(kind,p,split,i)
                    if tuple(q['identity']) not in blocked:break
                blocked.add(tuple(q['identity']))
                q['id']=q['id'].replace('v13_','v15_');q['question']=reword(q,split=='test')
                out[split].append(q)
                if split=='train' and kind=='moment' and i<12:
                    b=q['bindings'];prefix=f"X has mean {b['mean']} and variance {b['variance']}. Y={b['scale']}*X+{b['offset']}. "
                    for label,question,expr in (
                        ('mean','Set up E[Y].',f"{b['scale']}*{b['mean']}+{b['offset']}"),
                        ('variance','Set up Var(Y).',f"{b['scale']}**2*{b['variance']}")):
                        sub=dict(q,id=q['id']+'_'+label,category='moment_'+label,question=prefix+question,
                                 expression=expr,answer=calculate(expr),identity=q['identity']+['stage',label])
                        sub.pop('target',None);out[split].append(sub)
    return out


if __name__=='__main__':
    data=build();p=Path(__file__).resolve().parents[1]/'docs/STATS_V0_15_FROZEN_QUESTIONS.json'
    p.write_text(json.dumps(data,indent=2)+'\n');print(digest(data),{s:len(r) for s,r in data.items()})
