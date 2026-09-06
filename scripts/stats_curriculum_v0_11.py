"""Exact arithmetic curriculum; canonical identities and answer-disjoint splits."""
import math, random, itertools, re
from fractions import Fraction as F
from stats_curriculum_v0_9 import calculate, prompt, build as build9
from stats_curriculum_v0_10 import build as build10, identity, make as stat_make

KINDS=('multiply','add','integer_power','fraction_power','reduce')

def make(kind, p, uid, variant=False):
    if kind in ('multiply','add'):
        a,b,c,d=p; op='*' if kind=='multiply' else '+'
        f,g=F(a,b),F(c,d)
        key=(kind,*sorted((str(f),str(g))))
        scale=3 if variant else 1
        expr=f'({a*scale}/{b*scale}){op}({c*scale}/{d*scale})'
        question=f'Calculate {expr} exactly.'
        if kind=='multiply':
            chain=[expr,f'({a*scale}*{c*scale})/({b*scale}*{d*scale})',f'{a*c*scale*scale}/{b*d*scale*scale}']
            rule='Multiply the numerators and the denominators, then divide both by their greatest common divisor.'
        else:
            den=math.lcm(b*scale,d*scale); x=a*scale*(den//(b*scale)); y=c*scale*(den//(d*scale))
            chain=[expr,f'{x}/{den}+{y}/{den}',f'({x}+{y})/{den}',f'{x+y}/{den}']
            rule='Use a common denominator, add the numerators, then reduce the fraction.'
    elif kind in ('integer_power','fraction_power'):
        a,b,k=p; key=(kind,str(F(a,b)),k)
        scale=3 if variant and kind=='fraction_power' else 1
        base=str(a) if kind=='integer_power' else f'({a*scale}/{b*scale})'
        expr=f'{base}**{k}'
        question=f'Calculate {expr} exactly.' if not (variant and kind=='integer_power') else f'Find the exact product of {k} copies of {a}.'
        chain=[expr,'*'.join([base]*k)]
        if kind=='fraction_power':chain += [f'{(a*scale)**k}/{(b*scale)**k}']
        rule='A positive integer power means repeated multiplication; apply the power to numerator and denominator.'
    else:
        a,b,scale=p;key=(kind,str(F(a,b)))
        if variant:scale+=7
        top,bot=a*scale,b*scale; gcd=math.gcd(top,bot)
        expr=f'{top}/{bot}';question=f'Reduce {expr} to lowest terms.'
        chain=[expr,f'({top}/{gcd})/({bot}/{gcd})']
        rule='Divide numerator and denominator by their greatest common divisor.'
    answer=calculate(expr);chain.append(str(answer))
    chain=[s for i,s in enumerate(chain) if i==0 or s!=chain[i-1]]
    assert all(calculate(s)==answer for s in chain)
    return dict(id=uid,category=kind,parameters=list(p),identity=list(key),question=question,answer=str(answer),reference_expression=expr,reference_chain=chain,target='Formula: '+rule+'\nCalculation: '+' = '.join(chain)+'\nAnswer: '+str(answer))

def build():
    old9,old10=build9(),build10()
    # Block all earlier numerical answer values, including rehearsal and exposed tests.
    blocked={q['answer'] for data in (old9,old10) for rows in data.values() for q in rows}
    rng=random.Random(1111);seen=set();answers=set(blocked)
    data={s:[] for s in ('train','validation','test')}
    for split,count in (('train',80),('validation',8),('test',12)):
        for kind in KINDS:
            n=0;attempts=0
            while n<count:
                attempts+=1
                assert attempts<100000, (split,kind)
                b=rng.randint(7,60);a=rng.randint(1,b-1)
                cden=rng.randint(7,60);c=rng.randint(1,cden-1)
                f,g=F(a,b),F(c,cden)
                if kind in ('multiply','add'):p=(f.numerator,f.denominator,g.numerator,g.denominator)
                elif kind=='integer_power':p=(rng.randint(2,60),1,rng.randint(2,5))
                elif kind=='fraction_power':p=(f.numerator,f.denominator,rng.randint(2,5))
                else:p=(f.numerator,f.denominator,rng.randint(2,30))
                q=make(kind,p,f'v11_{split}_{kind}_{n:03d}');key=tuple(q['identity'])
                if key in seen or q['answer'] in answers:continue
                seen.add(key);answers.add(q['answer']);data[split].append(q);n+=1
    data['test_variants']=[make(q['category'],q['parameters'],q['id']+'_variant',True) for q in data['test']]
    oldids={identity(q) for old in (old9,old10) for rows in old.values() for q in rows if q['category'] in ('type_i','type_ii')}
    transfer=[]
    for topic in ('type_i','type_ii'):
        pool=([(n,k,a,d) for d in (60,80) for n in (3,4,5,6) for k in range(1,n) for a in range(1,d//2)] if topic=='type_i' else [(a,b,d) for d in (60,80) for a,b in itertools.combinations_with_replacement(range(1,d//2),2)])
        rng.shuffle(pool);count=0
        for p in pool:
            q=stat_make(topic,p,'test',len(transfer));qid=identity(q)
            if qid in oldids or q['answer'] in answers:continue
            q['id']=q['id'].replace('v10_','v11_transfer_');transfer.append(q);oldids.add(qid);answers.add(q['answer']);count+=1
            if count==24:break
        assert count==24
    data['transfer']=transfer
    validate(data)
    return data

def validate(data):
    assert {s:len(data[s]) for s in data}==dict(train=400,validation=40,test=60,test_variants=60,transfer=48)
    sets=[{tuple(q['identity']) for q in data[s]} for s in ('train','validation','test')]
    assert all(not a&b for a,b in itertools.combinations(sets,2))
    vals=[{q['answer'] for q in data[s]} for s in ('train','validation','test','transfer')]
    assert all(not a&b for a,b in itertools.combinations(vals,2))
    for a,b in zip(data['test'],data['test_variants']):
        assert a['identity']==b['identity'] and a['answer']==b['answer'] and a['question']!=b['question']
    for rows in data.values():
        for q in rows:assert all(calculate(s)==F(q['answer']) for s in q['reference_chain'])

def score(raw,expected,category=None):
    # Exact value only. Never evaluate an unfinished expression for answer credit.
    pattern=r'[+-]?\d+(?:/\d+|\.\d+)?'
    lines=re.findall(r'^\s*Answer:\s*('+pattern+r')\s*$',raw,re.M)
    final=lines[-1] if len(lines)==1 else None
    def valid(value):
        try:
            return F(value)==F(expected) and (category!='reduce' or value==str(F(value)))
        except (ValueError,ZeroDivisionError):return False
    strict=final is not None and valid(final)
    reviewed=strict
    if not lines:
        tail=raw.strip().splitlines()[-1] if raw.strip() else ''
        match=re.search(r'(?:^|=)\s*('+pattern+r')\s*$',tail)
        if match:reviewed=valid(match[1])
    return dict(correct=strict,reviewed_correct=reviewed,invalid=final is None,predicted=final or 'INVALID')

if __name__=='__main__':
    import json
    from stats_v0_3_common import digest
    d=build();print(json.dumps(dict(counts={k:len(v) for k,v in d.items()},sha256=digest(d))))
