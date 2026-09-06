"""Grounded formulation pilot: generated SFT, no new teacher calls.

Structural auto-credit is conservative. Numerically equal unmatched expressions
require review; they are NOT automatically declared mathematically wrong.
"""
import ast
import hashlib
import json
import math
import random
import re
from fractions import Fraction as F
from pathlib import Path
from exact_calculator import calculate

KINDS = ('poisson_time', 'poisson_scaled', 'moment', 'uniform_time',
         'binomial', 'exactly_one', 'at_least_one', 'interval')


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def make(kind, p, split, i):
    a,b,c,d = p
    if kind == 'poisson_time':
        rate,seconds=a,b
        question=f'A homogeneous Poisson arrival process averages {rate} arrivals per minute. Find the variance of the arrival count during {seconds} seconds.'
        bindings=dict(rate_per_minute=str(rate), duration_minutes=f'{seconds}/60')
        expression=f'{rate}*({seconds}/60)'
        answer=F(rate*seconds,60)
        identity=[kind,rate,seconds]
    elif kind == 'poisson_scaled':
        lam,scale,offset=a,c,d
        question=f'X is Poisson with mean {lam}. A recorded quantity is Y={scale}*X+{offset}. Find Var(Y).'
        bindings=dict(mean=str(lam), scale=str(scale), offset=str(offset))
        expression=f'{scale}**2*{lam}'
        answer=F(scale*scale*lam)
        identity=[kind,lam,scale,offset]
    elif kind == 'moment':
        mu,var,scale,offset=a,b,c,d
        question=f'A random variable X has mean {mu} and variance {var}. Find E[({scale}*X+{offset})^2].'
        bindings=dict(mean=str(mu), variance=str(var), scale=str(scale), offset=str(offset))
        expression=f'{scale}**2*{var}+({scale}*{mu}+{offset})**2'
        answer=F(scale*scale*var+(scale*mu+offset)**2)
        identity=[kind,mu,var,scale,offset]
    elif kind == 'uniform_time':
        upper=a; cutoff=b
        question=f'A waiting time T is uniform from 0 to {upper} minutes. Given that T exceeds {cutoff} seconds, find its conditional mean in minutes.'
        bindings=dict(upper_minutes=str(upper), cutoff_minutes=f'{cutoff}/60')
        expression=f'({upper}+{cutoff}/60)/2'
        answer=(F(upper)+F(cutoff,60))/2
        identity=[kind,upper,cutoff]
    elif kind == 'binomial':
        n,r,percent=c+2,d%c+1,a
        question=f'{n} independent tests each have a true null hypothesis. Each test rejects with probability {percent}%. Find the probability that exactly {r} reject.'
        bindings=dict(n=str(n), r=str(r), reject_probability=f'{percent}/100')
        expression=f'comb({n},{r})*({percent}/100)**{r}*(1-{percent}/100)**({n}-{r})'
        answer=F(math.comb(n,r))*F(percent,100)**r*(1-F(percent,100))**(n-r)
        identity=[kind,n,r,percent]
    elif kind in ('exactly_one','at_least_one'):
        miss1,miss2=a,b%39+11
        event='exactly one method detects' if kind=='exactly_one' else 'at least one method detects'
        question=f'Two independent methods inspect a defect. Method A misses it with probability {miss1}%, and method B misses it with probability {miss2}%. Find the probability that {event} the defect.'
        bindings=dict(miss_a=f'{miss1}/100', miss_b=f'{miss2}/100')
        expression=(f'(1-{miss1}/100)*({miss2}/100)+({miss1}/100)*(1-{miss2}/100)'
                    if kind=='exactly_one' else f'1-({miss1}/100)*({miss2}/100)')
        x,y=F(miss1,100),F(miss2,100)
        answer=(1-x)*y+x*(1-y) if kind=='exactly_one' else 1-x*y
        # Keep both event variants with these parameters in the SAME split.
        identity=['detection_pair',*sorted((miss1,miss2))]
    else:
        lower,upper,k=a,a+b,c
        question=f'A normal-theory confidence interval is [{lower}, {upper}]. Holding its center, confidence level and population standard deviation fixed, multiply the sample size by {k*k}. Find the new upper endpoint.'
        bindings=dict(lower=str(lower), upper=str(upper), width_divisor=str(k))
        expression=f'({lower}+{upper})/2+({upper}-{lower})/(2*{k})'
        answer=F(lower+upper,2)+F(upper-lower,2*k)
        identity=[kind,lower,upper,k]
    target='Bindings: '+'; '.join(f'{k}={v}' for k,v in bindings.items())+'\nExpression: '+expression
    return dict(id=f'v13_{split}_{kind}_{i:03d}', category=kind, identity=identity,
                question=question, bindings=bindings, expression=expression,
                answer=str(answer), target=target, provenance='procedural_reference')


def prompt(q):
    return (q['question']+'\nReturn exactly two lines.\nBindings: '+
            '; '.join(k+'=<number or fraction expression>' for k in q['bindings'])+
            '\nExpression: <one fully substituted numerical expression; no final answer>'+
            '\nUse integers, fractions, +, -, *, /, ** and comb(n,r). Keep conversion factors and operations unevaluated. No explanation.')


def canonical(expression):
    calculate(expression)  # validates the bounded, non-executable grammar
    def walk(node):
        if isinstance(node, ast.Constant): return ('number',str(node.value))
        if isinstance(node, ast.UnaryOp): return (type(node.op).__name__,walk(node.operand))
        if isinstance(node, ast.Call): return (node.func.id,*(walk(x) for x in node.args))
        if isinstance(node, ast.BinOp):
            op=type(node.op).__name__
            if op in ('Add','Mult'):
                def collect(n):
                    if isinstance(n,ast.BinOp) and type(n.op).__name__==op:
                        return collect(n.left)+collect(n.right)
                    return [walk(n)]
                return (op,*sorted(collect(node),key=repr))
            return (op,walk(node.left),walk(node.right))
        raise ValueError('Unsupported syntax')
    return walk(ast.parse(expression,mode='eval').body)


def score(raw,q):
    result=dict(bindings_correct=False, executable=False, numeric_correct=False,
                structure_verified=False, correct=False, review_required=False)
    try:
        match=re.fullmatch(r'\s*Bindings: ([^\n]+)\nExpression: ([^\n]+)\s*',raw)
        if not match: return dict(result,invalid=True)
        pairs=[x.strip().split('=',1) for x in match[1].split(';')]
        if any(len(x)!=2 for x in pairs): return dict(result,invalid=True)
        bindings=dict(pairs)
        if len(bindings)!=len(pairs): return dict(result,invalid=True)
        result['bindings_correct']=(bindings.keys()==q['bindings'].keys() and
            all(calculate(bindings[k].strip())==calculate(v) for k,v in q['bindings'].items()))
        expression=match[2].strip()
        result['computed']=calculate(expression)
        result['executable']=True
        result['numeric_correct']=result['computed']==q['answer']
        result['structure_verified']=canonical(expression)==canonical(q['expression'])
        result['correct']=all(result[k] for k in ('bindings_correct','numeric_correct','structure_verified'))
        result['review_required']=result['numeric_correct'] and not result['structure_verified']
        return dict(result,invalid=False)
    except (ValueError,SyntaxError,OverflowError,RecursionError):
        return dict(result,invalid=True)


def build():
    rng=random.Random(1313)
    data={s:[] for s in ('train','validation','test')}
    from stats_curriculum_v0_10 import identity as old_identity
    blocked=set()
    docs=Path(__file__).resolve().parents[1]/'docs'
    for version in (9,10,11,12):
        previous=json.loads((docs/f'STATS_V0_{version}_FROZEN_QUESTIONS.json').read_text())
        for rows in previous.values():
            for q in rows:
                category=q['category']
                if category in ('type_i','type_ii','type_i_pipeline','type_ii_pipeline'):
                    blocked.add(old_identity(dict(category=category.removesuffix('_pipeline'),parameters=q['parameters'])))
    owners={}; used=set()
    for split,count in (('train',48),('validation',8),('test',12)):
        for kind in KINDS:
            found=0
            for _ in range(100000):
                p=(rng.randrange(21,49),rng.randrange(61,599),rng.randrange(2,8),rng.randrange(2,19))
                q=make(kind,p,split,found)
                key=tuple(q['identity'])
                if kind=='binomial':
                    oldkey=('type_i',key[1],key[2],str(F(key[3],100)))
                    if oldkey in blocked: continue
                if kind in ('exactly_one','at_least_one'):
                    oldkey=('type_ii',*sorted(str(F(x,100)) for x in key[1:]))
                    if oldkey in blocked: continue
                if key in owners and owners[key]!=split: continue
                if (kind,key) in used: continue
                owners[key]=split;used.add((kind,key));data[split].append(q);found+=1
                if found==count: break
            assert found==count
    validate(data)
    return data


def validate(data):
    assert {s:len(v) for s,v in data.items()}==dict(train=384,validation=64,test=96)
    identities={s:{tuple(q['identity']) for q in rows} for s,rows in data.items()}
    assert not (identities['train']&identities['test'] or identities['train']&identities['validation'] or identities['test']&identities['validation'])
    for rows in data.values():
        for q in rows:
            assert calculate(q['expression'])==q['answer']
            assert score(q['target'],q)['correct']


if __name__=='__main__':
    data=build()
    path=Path(__file__).resolve().parents[1]/'docs/STATS_V0_13_FROZEN_QUESTIONS.json'
    path.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(counts={s:len(v) for s,v in data.items()},sha256=digest(data))))
