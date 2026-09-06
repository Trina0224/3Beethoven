"""Targeted short-calculation curriculum with disjoint parameter tuples."""
import ast,itertools,math,operator,random,re
from fractions import Fraction as F
from collections import Counter
from stats_v0_3_common import digest
TOPICS=('poisson','expectation','uniform','type_i','type_ii','confidence')

def calculate(expr):
    tree=ast.parse(expr.replace('^','**').replace('×','*').replace('÷','/').replace('²','**2').strip(),mode='eval')
    if len(list(ast.walk(tree)))>80:raise ValueError('Expression too large')
    ops={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.Pow:operator.pow}
    def walk(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float) and abs(n.value)<10**12:return F(str(n.value))
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return -walk(n.operand)
        if isinstance(n,ast.BinOp) and type(n.op) in ops:
            a,b=walk(n.left),walk(n.right)
            if isinstance(n.op,ast.Pow) and (b.denominator!=1 or abs(b)>8):raise ValueError('Power out of bounds')
            value=ops[type(n.op)](a,int(b) if isinstance(n.op,ast.Pow) else b)
            if value.numerator.bit_length()>4096 or value.denominator.bit_length()>4096:raise ValueError('Numeric result too large')
            return value
        raise ValueError('Unsupported expression')
    return walk(tree.body)

def pools(topic):
    if topic=='poisson':return [(s,k,a,b) for s in (0,1) for k in range(2,10) for a in (2,3,4) for b in ((0,) if s==0 else (1,5))]
    if topic=='expectation':return list(itertools.product(range(2,10),(1,2,3,4),(2,3),(1,2)))
    if topic=='uniform':return list(itertools.product(range(0,6),range(2,10)))
    if topic=='type_i':return [(n,r,p) for n in (3,4,5,6) for r in range(1,n) for p in (1,2,3,4,5,6,7,8,9)]
    if topic=='type_ii':return list(itertools.combinations_with_replacement(range(1,10),2))
    return list(itertools.product(range(20,61,5),(3,6,9,12),(2,3,4)))

def make(topic,params,split,i):
    context={'train':'workshop','validation':'exhibition','test':'research station'}[split]
    if topic=='poisson':
        s,k,a,b=params
        if s==0:
            q=f'A {context} counts arrivals from a Poisson process at rate {k} per minute. N is the total during {a} minutes. Find Var(N).'
            expr=f'{k}*{a}';rule='A Poisson count has variance equal to rate times duration.';wrong=[k,k*a*a,(k*a)**2]
        else:
            q=f'At a {context}, X is Poisson with mean {k}. A display reads Y={a}X+{b}, using that same random X. Find Var(Y).'
            expr=f'{a}**2*{k}';rule='Var(aX+b)=a^2 Var(X), and a Poisson variable has variance equal to its mean.';wrong=[a*k,a*a*k+b,k+b]
    elif topic=='expectation':
        mu,var,a,b=params
        q=f'A {context} measurement X has E[X]={mu} and Var(X)={var}. Define Y={a}X+{b}. Find E[Y^2].'
        expr=f'{a}**2*{var}+({a}*{mu}+{b})**2';rule='E[Y^2]=Var(Y)+E[Y]^2, with Var(aX+b)=a^2 Var(X) and E[aX+b]=aE[X]+b.';wrong=[(a*mu+b)**2,a*a*var,a*a*(mu*mu+var)+b*b]
    elif topic=='uniform':
        low,k=params;high=low+4*k;threshold=low+2*k
        q=f'A {context} waiting time T is continuously uniform on [{low},{high}]. Given T>{threshold}, find E[T | T>{threshold}].'
        expr=f'({threshold}+{high})/2';rule='Conditioning a continuous uniform variable above an interior threshold leaves a uniform interval from the threshold to the upper endpoint; its mean is their midpoint.';wrong=[threshold,high,k]
    elif topic=='type_i':
        n,r,pn=params;p=F(pn,20);c=math.comb(n,r)
        q=f'A {context} runs {n} independent tests when every null hypothesis is true. Each rejects its null with probability {p}. What is the probability exactly {r} tests reject?'
        expr=f'{c}*({p})**{r}*(1-({p}))**{n-r}';rule='The probability of exactly r successes in n independent trials is C(n,r)*p^r*(1-p)^(n-r).';wrong=[p**r*(1-p)**(n-r),p,1-p]
    elif topic=='type_ii':
        x,y=params;b1,b2=F(x,20),F(y,20)
        q=f'Two independent methods inspect an existing defect at a {context}. Their miss probabilities are {b1} and {b2}. What is the probability exactly one method detects it?'
        expr=f'(1-({b1}))*({b2})+({b1})*(1-({b2}))';rule='Exactly one detection is the sum of the disjoint detect/miss and miss/detect probabilities; independence permits products.';wrong=[(1-b1)*(1-b2),1-b1*b2,b1*b2]
    else:
        center,h,f=params
        q=f'A {context} reports a normal-theory mean confidence interval [{center-h},{center+h}]. Keep the center, confidence level and population standard deviation fixed; multiply sample size by {f*f}. Find the new upper endpoint.'
        expr=f'{center}+{h}/{f}';rule='The center is fixed and half-width is inversely proportional to sqrt(sample size); add the new half-width to the center.';wrong=[center+h,F(center)+F(h,f*f),(center+h)*f*f]
    ans=calculate(expr);choices=[]
    for v in wrong+[F(j,100) for j in range(1,100)]:
        v=F(v)
        if v!=ans and v not in choices:choices.append(v)
        if len(choices)==3:break
    pos=i%4;choices.insert(pos,ans)
    return dict(id=f'v09_{split}_{topic}_{i:03d}',category=topic,family=topic+'_targeted',parameters=list(params),question=q,choices=list(map(str,choices)),answer_letter='ABCD'[pos],answer=str(ans),reference_expression=expr,reference_rule=rule)

def build():
    out={k:[] for k in ('train','validation','test')}
    for ti,topic in enumerate(TOPICS):
        pool=pools(topic);random.Random(909+ti).shuffle(pool)
        start=0
        for split,n in (('train',30),('validation',4),('test',8)):
            for params in pool[start:start+n]:out[split].append(make(topic,params,split,len(out[split])))
            start+=n
    for split,n in (('train',180),('validation',24),('test',48)):
        assert len(out[split])==n
        assert Counter(r['answer_letter'] for r in out[split])==dict.fromkeys('ABCD',n//4)
    sets=[{(r['category'],tuple(r['parameters'])) for r in rows} for rows in out.values()]
    assert all(not a&b for a,b in itertools.combinations(sets,2))
    for rows in out.values():
        for r in rows:assert calculate(r['reference_expression'])==F(r['choices']['ABCD'.index(r['answer_letter'])])
    return out

def prompt(q):
    return q['question']+'\nGive a concise numerical solution in exactly three short lines.\nFormula: state the needed formula or operation.\nCalculation: substitute the numbers and calculate.\nAnswer: give only the final numerical value on this line.\nUse fractions when exact. No introduction or commentary.'

def numeric_score(raw,expected):
    from diagnose_stats_v0_7 import numeric
    value=numeric(raw);gold=F(expected)
    correct=value is not None and abs(float(value-gold))<=max(1e-10,abs(float(gold))*1e-4)
    return str(value) if value is not None else 'INVALID',correct,value is None

if __name__=='__main__':
    import json
    data=build();print(json.dumps(dict(counts={k:len(v) for k,v in data.items()},sha256=digest(data))))
