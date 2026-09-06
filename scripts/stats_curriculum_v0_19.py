"""Frozen full-skill replay repair; historical teacher targets stay separate."""
import ast,json,random,re
from pathlib import Path
from stats_curriculum_v0_13 import KINDS,make as old_make,digest
from stats_curriculum_v0_14 import reword
from stats_curriculum_v0_18 import make,TRACKS,score as base_score,prompt
from exact_calculator import calculate
DOCS=Path(__file__).resolve().parents[1]/'docs'
def score(raw,q):
    normalized=re.sub(r'\bC\s*\(', 'comb(',raw)
    g=base_score(normalized,q)
    g['original_raw_unchanged']=True
    g['comb_notation_normalized']=normalized!=raw
    if normalized!=raw:g['format_exact']=False
    if g.get('review_required') and q.get('track')=='second_moment' and q.get('depth')==2:
        n=q['parameters'][0]
        text=g.get('normalized_expression','')
        from stats_curriculum_v0_13 import canonical
        if any(canonical(text)==canonical(x) for x in (f'{n}*({n}+1)',f'{n}*(1+{n})')):
            g.update(correct=True,math_correct=True,review_required=False,reason='Frozen Poisson second-moment factorization.')
    g['grader_version']='v19-frozen-equivalences'
    return g

def keys(t,p):
    a,m,c,b,v,u,k=p
    return {'poisson_variance':[(a,),(a,m)],'scaled_variance':[(v,c,b),(a,c,b),(a,m,c,b)],'second_moment':[(a,),(a,v),(a,m)],'conditional_wait':[(u,k)]}[t]

def build():
    old=json.loads((DOCS/'STATS_V0_19_REPLAY_SOURCE.json').read_text())
    old=[r for r in old if r['category'] in KINDS]
    assert len(old)==192 and all(sum(r['category']==k for r in old)==24 for k in KINDS)
    assert all(r.get('teacher_raw') for r in old)
    data18=json.loads((DOCS/'STATS_V0_18_FROZEN_QUESTIONS.json').read_text())
    train=old+[dict(source_id=q['id'],category=q['category'],prompt=prompt(q),target=q['target'],source='procedural_reference') for q in data18['train']]
    blocked=set()
    for v in (13,14,15,16,17):
        for rows in json.loads((DOCS/f'STATS_V0_{v}_FROZEN_QUESTIONS.json').read_text()).values():
            for q in rows:
                ident=q['identity'];blocked.add(tuple(ident[:ident.index('stage')] if 'stage' in ident else ident))
    for name in ('TEACHER_PERTURBATIONS','TEACHER_SCAFFOLD'):
        for pair in json.loads((DOCS/f'STATS_V0_14_{name}.json').read_text())['pairs']:
            blocked.update(tuple(q['identity']) for q in pair['questions'])
    used={t:set() for t in TRACKS}
    for rows in data18.values():
        for q in rows:
            p=q['parameters'];used[q['track']].update(keys(q['track'],p))
            a,m,c,b,v,u,k=p
            blocked.update([('poisson_scaled',a,c,b),('poisson_time',a,m*60+30),('uniform_time',u,k*60+30)])
    rng=random.Random(1919);oldtest=[];newtest=[]
    for kind in KINDS:
        for i in range(12):
            for _ in range(100000):
                q=old_make(kind,[rng.randrange(12,69),rng.randrange(61,601),rng.randrange(2,8),rng.randrange(2,24)],'test',i)
                if tuple(q['identity']) not in blocked:break
            else:raise RuntimeError('Exhausted identities')
            blocked.add(tuple(q['identity']));q['id']=q['id'].replace('v13_','v19_old_');q['question']=reword(q,True);oldtest.append(q)
    for t in TRACKS:
        for i in range(8):
            for _ in range(100000):
                p=[rng.randrange(13,98),rng.randrange(2,15),rng.randrange(2,9),rng.randrange(3,29),rng.randrange(17,299),rng.randrange(22,89),rng.randrange(2,19)]
                a,m,c,b,v,u,k=p
                oldkeys=[('poisson_scaled',a,c,b),('poisson_time',a,m*60+30),('uniform_time',u,k*60+30)]
                if not any(x in used[t] for x in keys(t,p)) and not any(x in blocked for x in oldkeys):break
            else:raise RuntimeError('Exhausted primitive identities')
            used[t].update(keys(t,p));blocked.update(oldkeys)
            for d in (1,2,3):
                q=make(t,d,p,'test',i);q['id']=q['id'].replace('v18_','v19_');q['story_id']=q['story_id'].replace('v18_','v19_');newtest.append(q)
    out=dict(train=train,old_validation=json.loads((DOCS/'STATS_V0_17_FROZEN_QUESTIONS.json').read_text())['validation'],new_validation=data18['validation'],old_test=oldtest,new_test=newtest)
    assert len(train)==480 and len({r['source_id'] for r in train})==480
    for name,qs in out.items():
        if name=='train':continue
        for q in qs:
            assert calculate(q['expression'])==q['answer']
            assert score('Expression: '+q['expression'],q)['correct'],q['id']
    return out
if __name__=='__main__':
    d=build();(DOCS/'STATS_V0_19_FROZEN_QUESTIONS.json').write_text(json.dumps(d,indent=2)+'\n');print(digest(d))
