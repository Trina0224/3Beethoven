"""Fraction-execution follow-up: exact references and unseen parameter tuples."""
import itertools,math,random
from fractions import Fraction as F
from stats_curriculum_v0_9 import build as old_build,calculate,prompt,numeric_score

def identity(q):
    p=q['parameters']
    if q['category']=='type_i':
        return ('type_i',p[0],p[1],str(F(p[2],p[3] if len(p)>3 else 20)))
    return ('type_ii',*sorted((str(F(p[0],p[2] if len(p)>2 else 20)),str(F(p[1],p[2] if len(p)>2 else 20)))))

def make(topic,p,split,i):
    if topic=='type_i':
        n,r,a,d=p;b=d-a;c=math.comb(n,r)
        expr=f'{c}*({a}/{d})**{r}*({b}/{d})**{n-r}'
        top=c*a**r*b**(n-r);bottom=d**n
        chain=[expr,f'{c}*({a**r}/{d**r})*({b**(n-r)}/{d**(n-r)})',f'({c}*{a**r}*{b**(n-r)})/({d**r}*{d**(n-r)})',f'{top}/{bottom}']
        question=f'A laboratory runs {n} independent tests when all null hypotheses are true. Each rejects with probability {F(a,d)}. What is the probability exactly {r} reject?'
        wrong=[F(a,d),1-F(a,d),F(top,bottom)/c]
    else:
        a,b,d=p
        expr=f'(1-({a}/{d}))*({b}/{d})+({a}/{d})*(1-({b}/{d}))'
        x,y=(d-a)*b,a*(d-b);top=x+y;bottom=d*d
        chain=[expr,f'({d-a}/{d})*({b}/{d})+({a}/{d})*({d-b}/{d})',f'{x}/{bottom}+{y}/{bottom}',f'({x}+{y})/{bottom}',f'{top}/{bottom}']
        question=f'Two independent methods inspect an existing defect. Their miss probabilities are {F(a,d)} and {F(b,d)}. What is the probability exactly one detects it?'
        wrong=[F((d-a)*(d-b),d*d),1-F(a*b,d*d),F(a*b,d*d)]
    ans=F(top,bottom);chain.append(str(ans))
    choices=[]
    for v in wrong+[F(j,100) for j in range(1,100)]:
        if v!=ans and v not in choices:choices.append(v)
        if len(choices)==3:break
    choices.insert(i%4,ans)
    assert all(calculate(x)==ans for x in chain)
    return dict(id=f'v10_{split}_{topic}_{i:03d}',category=topic,family=topic+'_fraction',parameters=list(p),question=question,choices=list(map(str,choices)),answer_letter='ABCD'[i%4],answer=str(ans),reference_expression=expr,reference_chain=chain)

def build():
    old={identity(q) for rows in old_build().values() for q in rows if q['category'] in ('type_i','type_ii')}
    out={s:[] for s in ('train','validation','test')}
    for ti,topic in enumerate(('type_i','type_ii')):
        pool=([(n,r,a,40) for n in (3,4,5,6) for r in range(1,n) for a in range(1,20)] if topic=='type_i' else [(a,b,40) for a,b in itertools.combinations_with_replacement(range(1,20),2)])
        random.Random(1010+ti).shuffle(pool)
        pool=[p for p in pool if identity(dict(category=topic,parameters=p)) not in old]
        start=0
        for s,n in (('train',48),('validation',8),('test',24)):
            for p in pool[start:start+n]:out[s].append(make(topic,p,s,len(out[s])))
            start+=n
    groups=[{identity(q) for q in rows} for rows in out.values()]
    assert all(not a&b for a,b in itertools.combinations(groups,2)) and all(not x&old for x in groups)
    return out

if __name__=='__main__':
    import json
    from stats_v0_3_common import digest
    d=build();print(json.dumps({'counts':{k:len(v) for k,v in d.items()},'sha256':digest(d)}))
