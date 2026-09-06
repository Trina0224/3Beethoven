"""Frozen balanced selection and untouched test for adapter interpolation/rehearsal."""
import json, random
from pathlib import Path
from stats_curriculum_v0_13 import KINDS,make,digest
from stats_curriculum_v0_14 import reword
from prepare_verified_distillation_v0_14 import formulation_prompt as prompt
from stats_curriculum_v0_15 import score as original_score
from formulation_grader import parse_expression,shape


def score(raw,q):
    g=original_score(raw,q)
    if g['math_correct'] is None and g['executable'] and g.get('computed')==q['answer'] and q['category'] in ('moment','poisson_scaled'):
        a=q['bindings']['scale'];alt=q['expression'].replace(a+'**2',str(int(a)**2),1)
        if shape(parse_expression(g['normalized_expression'],{}))==shape(parse_expression(alt,{})):
            g.update(math_correct=True,correct=True,review_required=False,reason='Pre-frozen evaluated scale-square equivalence; remaining structure matches.')
    return dict(g,grader_version='v17-frozen-prior-equivalences')


def build():
    docs=Path(__file__).resolve().parents[1]/'docs';blocked=set()
    for v in (13,14,15,16):
        old=json.loads((docs/f'STATS_V0_{v}_FROZEN_QUESTIONS.json').read_text())
        blocked.update(tuple(q['identity'][:5] if 'stage' in q['identity'] else q['identity']) for rows in old.values() for q in rows)
    for name in ('TEACHER_PERTURBATIONS','TEACHER_SCAFFOLD'):
        old=json.loads((docs/f'STATS_V0_14_{name}.json').read_text())
        blocked.update(tuple(q['identity']) for pair in old['pairs'] for q in pair['questions'])
    rng=random.Random(1717);out={s:[] for s in ('validation','test')}
    for split,n in (('validation',6),('test',12)):
        for kind in KINDS:
            for i in range(n):
                for _ in range(100000):
                    q=make(kind,[rng.randrange(12,69),rng.randrange(61,601),rng.randrange(2,8),rng.randrange(2,24)],split,i)
                    if tuple(q['identity']) not in blocked:break
                else:raise RuntimeError('No fresh identity')
                blocked.add(tuple(q['identity']));q['id']=q['id'].replace('v13_','v17_')
                q['question']=reword(q,True);q.pop('target',None);out[split].append(q)
    return out


def selection_key(metrics):
    # Equal-sized families: total is macro accuracy. Tie: non-target retention,
    # then weakest-family performance. Caller resolves remaining ties by order.
    cats=metrics['by_category'];return (metrics['correct'],sum(v['correct'] for c,v in cats.items() if c not in ('moment','poisson_scaled')),min(v['correct'] for v in cats.values()))

if __name__=='__main__':
    d=build();p=Path(__file__).resolve().parents[1]/'docs/STATS_V0_17_FROZEN_QUESTIONS.json';p.write_text(json.dumps(d,indent=2)+'\n');print(digest(d))
