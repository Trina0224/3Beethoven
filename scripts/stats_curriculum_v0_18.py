"""One, two and three explicitly enumerated conceptual transformations.

New targets are procedural supervised references, NOT claimed teacher responses.
All variants of a parameterized story stay in one split.
"""
import ast, json, random
from fractions import Fraction as F
from pathlib import Path
from exact_calculator import calculate
from stats_curriculum_v0_13 import digest, canonical
from formulation_grader import grade, parse_expression
from prepare_verified_distillation_v0_14 import formulation_prompt as prompt

TRACKS=('poisson_variance','scaled_variance','second_moment','conditional_wait')

def make(track,depth,p,split,i):
    lam,minutes,scale,offset,variance,upper,cut=p
    seconds=minutes*60+30
    if track=='poisson_variance':
        if depth==1:
            question=f'X is a Poisson count with mean {lam}. Find Var(X).'
            expr=str(lam);rules=['Poisson mean to variance']
        elif depth==2:
            question=f'A homogeneous Poisson process has rate {lam} arrivals per minute. Find the count variance over {minutes} minutes.'
            expr=f'{lam}*{minutes}';rules=['rate and duration to count mean','Poisson mean to variance']
        else:
            question=f'A homogeneous Poisson process has rate {lam} arrivals per minute. Find the count variance over {seconds} seconds.'
            expr=f'{lam}*({seconds}/60)';rules=['seconds to minutes','rate and duration to count mean','Poisson mean to variance']
    elif track=='scaled_variance':
        if depth==1:
            question=f'X has variance {variance}. Y={scale}*X+{offset}. Find Var(Y).'
            expr=f'{scale}**2*{variance}';rules=['affine variance transformation']
        elif depth==2:
            question=f'X is Poisson with mean {lam}. Y={scale}*X+{offset}. Find Var(Y).'
            expr=f'{scale}**2*{lam}';rules=['Poisson mean to variance','affine variance transformation']
        else:
            question=f'A homogeneous Poisson process has rate {lam} arrivals per minute. X counts arrivals over {minutes} minutes. Y={scale}*X+{offset}. Find Var(Y).'
            expr=f'{scale}**2*({lam}*{minutes})';rules=['rate and duration to count mean','Poisson mean to variance','affine variance transformation']
    elif track=='second_moment':
        if depth==1:
            question=f'X has mean {lam} and variance {variance}. Find E[X**2].'
            expr=f'{variance}+{lam}**2';rules=['mean and variance to second moment']
        elif depth==2:
            question=f'X is Poisson with mean {lam}. Find E[X**2].'
            expr=f'{lam}+{lam}**2';rules=['Poisson mean to variance','mean and variance to second moment']
        else:
            question=f'A homogeneous Poisson process has rate {lam} arrivals per minute. X counts arrivals over {minutes} minutes. Find E[X**2].'
            expr=f'({lam}*{minutes})+({lam}*{minutes})**2';rules=['rate and duration to count mean','Poisson mean to variance','mean and variance to second moment']
    else:
        if depth==1:
            question=f'T is uniform on [{cut}, {upper}] minutes. Find E[T] in minutes.'
            expr=f'({cut}+{upper})/2';rules=['uniform bounds to mean']
        elif depth==2:
            question=f'T is uniform on [0, {upper}] minutes. Given T>{cut} minutes, find its conditional total mean in minutes.'
            expr=f'({cut}+{upper})/2';rules=['conditioning uniform support','uniform bounds to mean']
        else:
            question=f'T is uniform on [0, {upper}] minutes. Given T>{cut*60+30} seconds, find its conditional total mean in minutes.'
            expr=f'({cut*60+30}/60+{upper})/2';rules=['seconds to minutes','conditioning uniform support','uniform bounds to mean']
    q=dict(id=f'v18_{split}_{track}_{i:03d}_d{depth}',category='v18_'+track,track=track,depth=depth,
           story_id=f'v18_{split}_{track}_{i:03d}',parameters=p,question=question,bindings={},
           expression=expr,answer=calculate(expr),conceptual_rules=rules,
           provenance='procedural_reference; no teacher call')
    assert len(rules)==depth
    q['target']='Expression: '+expr
    return q

def score(raw,q):
    g=grade(raw,q)
    if g['math_correct'] is None and g['executable'] and g['computed']==q['answer']:
        # Arithmetic reassociation/commutation only: a*b/60 equals a*(b/60).
        # Never accept a number just because it equals the answer.
        def norm(node):
            if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Mult,ast.Div)):
                num=[];den=[]
                def collect(x,invert=False):
                    if isinstance(x,ast.BinOp) and isinstance(x.op,ast.Mult):collect(x.left,invert);collect(x.right,invert)
                    elif isinstance(x,ast.BinOp) and isinstance(x.op,ast.Div):collect(x.left,invert);collect(x.right,not invert)
                    else:(den if invert else num).append(norm(x))
                collect(node);return ('ratio',tuple(sorted(num,key=repr)),tuple(sorted(den,key=repr)))
            if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add):
                def terms(x):return terms(x.left)+terms(x.right) if isinstance(x,ast.BinOp) and isinstance(x.op,ast.Add) else [norm(x)]
                return ('sum',tuple(sorted(terms(node),key=repr)))
            if isinstance(node,ast.BinOp):return (type(node.op).__name__,norm(node.left),norm(node.right))
            return canonical(ast.unparse(node))
        if norm(parse_expression(g['normalized_expression'],{}))==norm(parse_expression(q['expression'],{})):
            g.update(math_correct=True,review_required=False,reason='Frozen commutative/reassociated arithmetic structure with exact agreement.')
    return dict(g,correct=g['math_correct'] is True,grader_version='v18-frozen')

def build():
    rng=random.Random(1818);data={s:[] for s in ('train','validation','test')};used={t:set() for t in TRACKS}
    # Block all previous numerical identities of overlapping full tasks.
    docs=Path(__file__).resolve().parents[1]/'docs';blocked=set()
    for v in (13,14,15,16,17):
        for rows in json.loads((docs/f'STATS_V0_{v}_FROZEN_QUESTIONS.json').read_text()).values():
            for q in rows:
                ident=q['identity'];blocked.add(tuple(ident[:ident.index('stage')] if 'stage' in ident else ident))
    for split,n in (('train',24),('validation',4),('test',8)):
        for track in TRACKS:
            for i in range(n):
                while True:
                    p=[rng.randrange(13,98),rng.randrange(2,15),rng.randrange(2,9),rng.randrange(3,29),rng.randrange(17,299),rng.randrange(22,89),rng.randrange(2,19)]
                    a,m,c,b,v,u,t=p
                    # Block repeated primitive identities too, not only full stories.
                    keys={'poisson_variance':[(a,),(a,m)],'scaled_variance':[(v,c,b),(a,c,b),(a,m,c,b)],'second_moment':[(a,),(a,v),(a,m)],'conditional_wait':[(u,t)]}[track]
                    old=[('poisson_scaled',a,c,b),('poisson_time',a,m*60+30),('uniform_time',u,t*60+30)]
                    if any(k in used[track] for k in keys) or any(k in blocked for k in old):continue
                    used[track].update(keys);break
                data[split].extend(make(track,d,p,split,i) for d in (1,2,3))
    validate(data);return data

def stage_rows(data,stage):
    rows=[]
    for depth in range(1,stage+1):
        for track in TRACKS:
            pool=[q for q in data['train'] if q['depth']==depth and q['track']==track]
            rows+=pool if depth==stage else pool[:12]
    return [dict(source_id=q['id'],prompt=prompt(q),target=q['target'],depth=q['depth']) for q in rows]

def validate(data):
    assert {s:len(qs) for s,qs in data.items()}==dict(train=288,validation=48,test=96)
    owners={}
    for split,qs in data.items():
        for q in qs:
            key=(q['track'],tuple(q['parameters']))
            assert owners.setdefault(key,split)==split
            assert len(q['conceptual_rules'])==q['depth']
            assert calculate(q['expression'])==q['answer'] and score(q['target'],q)['correct']
            assert not score('Expression: '+str(F(q['answer'])+1),q)['correct']
    assert [len(stage_rows(data,s)) for s in (1,2,3)]==[96,144,192]

if __name__=='__main__':
    data=build();path=Path(__file__).resolve().parents[1]/'docs/STATS_V0_18_FROZEN_QUESTIONS.json'
    path.write_text(json.dumps(data,indent=2)+'\n');print(digest(data))
