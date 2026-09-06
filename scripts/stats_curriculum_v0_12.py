"""Procedural arithmetic curriculum for exact statistical calculations."""
import itertools, math, random, re
from fractions import Fraction as F
from stats_curriculum_v0_9 import build as build9
from stats_curriculum_v0_10 import build as build10, identity
from stats_curriculum_v0_11 import build as build11
from stats_v0_3_common import digest

MICRO = ('multiply_steps', 'power_steps', 'gcd_steps', 'reduce_steps')

def prompt(q):
    return (q['question'] + '\nShow short exact arithmetic steps. End with exactly one line in the form '
            "Answer: <final value>. Use a reduced fraction when the answer is fractional. No introduction.")

def make_micro(kind, p, split, i):
    qid=f'v12_{split}_{kind}_{i:03d}'
    if kind=='multiply_steps':
        a,b=p; tens=(b//10)*10; ones=b%10; ans=a*b
        question=f'Calculate {a}*{b} exactly.'
        target=f'Method: split {b} into {tens}+{ones}.\nCalculation: {a}*{tens}={a*tens}; {a}*{ones}={a*ones}; {a*tens}+{a*ones}={ans}.\nAnswer: {ans}'
    elif kind=='power_steps':
        a,k=p; vals=[a]
        for _ in range(1,k): vals.append(vals[-1]*a)
        ans=vals[-1]; pieces=[f'{a}^2={vals[1]}']+[f'{a}^{j}={vals[j-2]}*{a}={vals[j-1]}' for j in range(3,k+1)]
        question=f'Calculate {a}^{k} exactly.'
        target='Method: multiply by the base once per power.\nCalculation: '+'; '.join(pieces)+f'.\nAnswer: {ans}'
    elif kind=='gcd_steps':
        a,b=p; x,y=max(a,b),min(a,b); steps=[]
        while y:
            steps.append(f'{x}={x//y}*{y}+{x%y}');x,y=y,x%y
        ans=x;question=f'Find gcd({a},{b}) using the Euclidean algorithm.'
        target='Method: repeat division with remainder.\nCalculation: '+'; '.join(steps)+f'.\nAnswer: {ans}'
    else:
        a,b=p;g=math.gcd(a,b);ans=F(a,b)
        question=f'Reduce {a}/{b} to lowest terms. Show how you find the gcd.'
        x,y=max(a,b),min(a,b);steps=[]
        while y:
            steps.append(f'{x}={x//y}*{y}+{x%y}');x,y=y,x%y
        target='GCD: '+'; '.join(steps)+f', so gcd={g}.\nCalculation: ({a}/{g})/({b}/{g})={ans}.\nAnswer: {ans}'
    return dict(id=qid,category=kind,parameters=list(p),identity=[kind,*p],question=question,answer=str(ans),target=target)

def make_stats(kind,p,split,i):
    qid=f'v12_{split}_{kind}_{i:03d}'
    if kind=='type_i_pipeline':
        n,r,a,d=p;c=math.comb(n,r);q=d-a
        ap=a**r;qp=q**(n-r);num=c*ap*qp;den=d**n;g=math.gcd(num,den);ans=F(num,den)
        question=f'{n} independent tests have true null hypotheses and each rejects with probability {F(a,d)}. What is the probability exactly {r} reject?'
        target=(f'Formula: C({n},{r})*({a}/{d})^{r}*({q}/{d})^{n-r}.\n'
                f'Powers: {a}^{r}={ap}; {q}^{n-r}={qp}; {d}^{n}={den}.\n'
                f'Multiply: C({n},{r})={c}; numerator={c}*{ap}*{qp}={num}.\n'
                f'Reduce: gcd({num},{den})={g}; ({num}/{g})/({den}/{g})={ans}.\nAnswer: {ans}')
    else:
        a,b,d=p;x=(d-a)*b;y=a*(d-b);num=x+y;den=d*d;g=math.gcd(num,den);ans=F(num,den)
        question=f'Two independent methods inspect a defect. Their miss probabilities are {F(a,d)} and {F(b,d)}. What is the probability exactly one detects it?'
        target=(f'Formula: (({d}-{a})/{d})*({b}/{d})+({a}/{d})*(({d}-{b})/{d}).\n'
                f'Products: ({d-a})*{b}={x}; {a}*({d-b})={y}; denominator={d}^2={den}.\n'
                f'Add: numerator={x}+{y}={num}.\n'
                f'Reduce: gcd({num},{den})={g}; ({num}/{g})/({den}/{g})={ans}.\nAnswer: {ans}')
    return dict(id=qid,category=kind,parameters=list(p),identity=[kind,*p],question=question,answer=str(ans),target=target)

def _take(pool, counts, make, rng, blocked_answers):
    rng.shuffle(pool);out={s:[] for s in counts};used=set();at=0
    for split,count in counts.items():
        while len(out[split])<count:
            assert at<len(pool);p=pool[at];at+=1
            q=make(p,split,len(out[split]));key=tuple(q['identity'])
            if key in used or q['answer'] in blocked_answers: continue
            used.add(key);blocked_answers.add(q['answer']);out[split].append(q)
    return out

def build():
    previous=(build9(),build10(),build11())
    blocked={q.get('answer') for data in previous for rows in data.values() for q in rows if q.get('answer') is not None}
    rng=random.Random(1212);data={s:[] for s in ('train','validation','micro_test','transfer_test')}
    pools={
      'multiply_steps':[(a,b) for a in range(113,998,7) for b in range(12,99,3) if b%10],
      'power_steps':[(a,k) for a in range(7,121) for k in (2,3,4,5)],
      'gcd_steps':[(g*x,g*y) for g in range(7,601) for x in range(2,45) for y in range(x+1,51) if math.gcd(x,y)==1],
      'reduce_steps':[(g*x,g*y) for g in range(7,100) for x in range(2,90) for y in range(x+1,101) if math.gcd(x,y)==1],
    }
    for kind in MICRO:
        chunks=_take(pools[kind],{'train':100,'validation':12,'micro_test':20},lambda p,s,i:make_micro(kind,p,s,i),rng,blocked)
        for s,rows in chunks.items():data[s].extend(rows)
    old_stats={identity(q) for old in previous for rows in old.values() for q in rows if q['category'] in ('type_i','type_ii')}
    specs=(('type_i_pipeline',[(n,r,a,d) for d in (50,70,90,100) for n in range(3,7) for r in range(1,n) for a in range(2,d//2)]),
           ('type_ii_pipeline',[(a,b,d) for d in (50,70,90,100) for a,b in itertools.combinations_with_replacement(range(2,d//2),2)]))
    for kind,pool in specs:
        rng.shuffle(pool);chosen={'train':[],'validation':[],'transfer_test':[]}
        needs={'train':80,'validation':8,'transfer_test':24}
        for p in pool:
            split='transfer_test' if p[-1] in (70,90) else ('train' if len(chosen['train'])<80 else 'validation')
            if len(chosen[split])>=needs[split]:continue
            basecat='type_i' if kind.startswith('type_i') else 'type_ii'
            if identity(dict(category=basecat,parameters=list(p))) in old_stats:continue
            q=make_stats(kind,p,split,len(chosen[split]))
            if q['answer'] in blocked:continue
            blocked.add(q['answer']);chosen[split].append(q)
            if all(len(chosen[s])==needs[s] for s in needs):break
        assert all(len(chosen[s])==needs[s] for s in needs),(kind,{s:len(v) for s,v in chosen.items()})
        for s,rows in chosen.items():data[s].extend(rows)
    validate(data);return data

def validate(data):
    assert {s:len(v) for s,v in data.items()}==dict(train=560,validation=64,micro_test=80,transfer_test=48)
    groups=[{tuple(q['identity']) for q in data[s]} for s in data]
    assert all(not a&b for a,b in itertools.combinations(groups,2))
    answers=[{q['answer'] for q in data[s]} for s in data]
    assert all(not a&b for a,b in itertools.combinations(answers,2))
    for rows in data.values():
        for q in rows:
            assert re.search(r'^Answer: '+re.escape(q['answer'])+r'$',q['target'],re.M)

def score(raw,expected,category=None):
    pattern=r'[+-]?\d+(?:/\d+|\.\d+)?';lines=re.findall(r'^\s*Answer:\s*('+pattern+r')\s*$',raw,re.M)
    final=lines[-1] if len(lines)==1 else None
    try: correct=final is not None and F(final)==F(expected) and ('/' not in final or final==str(F(final)))
    except (ValueError,ZeroDivisionError):correct=False
    return dict(correct=correct,invalid=final is None,predicted=final or 'INVALID')

if __name__=='__main__':
    import json
    d=build();print(json.dumps(dict(counts={k:len(v) for k,v in d.items()},sha256=digest(d))))
