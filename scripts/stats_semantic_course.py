"""Matched-row curriculum comparison; authored supervision, zero teacher calls."""
import argparse
import copy
import json
import random
import zipfile
from collections import Counter, defaultdict
from fractions import Fraction as F
from pathlib import Path
from exact_calculator import calculate
from stats_consolidation_semantics import oracle, task_key, validate_question
from stats_consolidation_pilot import digest
from stats_expansion_data import walk
from stats_multiview_corpus import SUFFIX

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ('neither', 'both', 'exactly_one', 'at_least_one', 'same')
MOMENTS = ('mean', 'variance', 'second_moment')
TRAIN_CONTEXTS = (('switches', 'activate'), ('servers', 'respond'),
                  ('sensors', 'detect the signal'), ('components', 'work'))
TEST_CONTEXTS = (('alarms', 'sound'), ('transmitters', 'deliver a packet'))


def frozen(x):
    return tuple(frozen(v) for v in x) if isinstance(x, (list, tuple)) else x


def event(params, target, split, i, view):
    n, m, den, negative = params
    p, z = F(n, den), F(m, den)
    s = dict(kind='events', target=target, p=str(p), p_b=str(z))
    a, b = (den-n, den-m) if negative else (n, m)
    def shown(v):
        return f'{v}%' if den == 100 else f'{v}/{den}'
    pa = f'(1-({a}/{den}))' if negative else f'({a}/{den})'
    pb = f'(1-({b}/{den}))' if negative else f'({b}/{den})'
    expression = {'both':f'{pa}*{pb}', 'neither':f'(1-{pa})*(1-{pb})',
        'exactly_one':f'{pa}*(1-{pb})+(1-{pa})*{pb}',
        'at_least_one':f'1-(1-{pa})*(1-{pb})',
        'same':f'{pa}*{pb}+(1-{pa})*(1-{pb})'}[target]
    contexts = TRAIN_CONTEXTS if split == 'train' else TEST_CONTEXTS
    noun, verb = contexts[(i//4) % len(contexts)]
    facts = (f'Two independent {noun}, A and B, each either {verb} or do not. '
             f'The probabilities that they {"do not " if negative else ""}{verb} '
             f'are {shown(a)} and {shown(b)}, respectively. ')
    if negative and den == 100:
        facts += f'Here a miss means failure to {verb}. '
    direct = {'both':f'both {verb}', 'neither':f'neither of them {verb}',
        'exactly_one':f'exactly one of them {verb}',
        'at_least_one':f'at least one of them {verb}',
        'same':'their two outcomes are the same (both affirmative or both negative)'}
    expanded = {'both':f'A does {verb} and B does {verb}',
        'neither':f'A does not {verb} and B does not {verb}',
        'exactly_one':f'A does {verb} while B does not, or B does {verb} while A does not',
        'at_least_one':f'it is not the case that both fail to {verb}',
        'same':'either both outcomes are affirmative or both outcomes are negative'}
    reserved = {'both': 'the number of affirmative outcomes is two',
        'neither':'there are zero affirmative outcomes',
        'exactly_one':'the number of affirmative outcomes is one, excluding zero and two',
        'at_least_one':'the number of affirmative outcomes is nonzero',
        'same':'the number of affirmative outcomes is even'}
    clause = direct[target] if view == 0 else expanded[target] if split == 'train' else reserved[target]
    question = facts+f'What is the probability that {clause}?'
    return make_question(s, expression, question, split, 'events', i, view, target,
                         representation='percentage' if den == 100 else 'fraction', negative_input=negative, context=noun)


def moment(params, target, split, i, view):
    m, v, a, b = params
    s = dict(kind='moments', target=target, mean=str(m), variance=str(v), scale=str(a), offset=str(b))
    expression = {'mean':f'{a}*{m}+{b}', 'variance':f'{a}**2*{v}',
        'second_moment':f'{a}**2*{v}+({a}*{m}+{b})**2'}[target]
    facts = f'X has mean {m} and variance {v}. '
    if view == 0:
        facts += f'Define Y={a}*X+{b}. '
        question = facts+{'mean':'Find the expected value of Y.',
            'variance':'Find the variance of Y.',
            'second_moment':'Find E[Y**2], the expectation after squaring Y.'}[target]
    elif split == 'train':
        facts += f'Each observation of X is multiplied by {a}, then increased by {b}. '
        question = facts+{'mean':'Find the average of these transformed observations, without squaring them.',
            'variance':'Find the variance of these transformed observations, not their average.',
            'second_moment':'Square each transformed observation and then average those squares. '
                            'Find that expectation, not the square of the average.'}[target]
    else:
        facts += f'A recorded measurement is obtained by taking {a} times X and adding {b}. '
        question = facts+{'mean':'Average the recorded measurements themselves. Set up the resulting expectation.',
            'variance':'Set up the mean squared deviation of recorded measurements from their own mean.',
            'second_moment':'For every recorded measurement, multiply that measurement by itself. '
                            'Set up the average of the resulting products.'}[target]
    return make_question(s, expression, question, split, 'moments', i, view, target,
                         representation='symbolic' if view == 0 else 'verbal')


def make_question(s, expression, question, split, domain, i, view, target, **metadata):
    sid = f'semantic_{split}_{domain}_{i:02}'
    q = dict(id=f'{sid}_{target}_v{view}', story_id=sid,
             category=target if domain == 'events' else 'moment_'+target,
             semantics=s, bindings={}, expression=expression, answer=str(oracle(s)),
             question=question, domain=domain, view='familiar' if view == 0 else 'verbal', **metadata)
    validate_question(q)
    return q


def generate(archive, corrected, output):
    base = json.loads(corrected.read_text())['train']
    assert len(base) == 750
    blocked = {frozen(r['semantic_key']) for r in base}
    # Scan all accessible historical question sets, including V56 source files.
    documents = []
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if name.startswith('executed_source/docs/') and name.endswith('.json'):
                documents.append(json.loads(z.read(name)))
    for path in (ROOT/'docs').glob('*.json'):
        if path.name.startswith('STATS_SEMANTIC_COURSE'): continue
        try: documents.append(json.loads(path.read_text()))
        except (ValueError, UnicodeError): pass
    for doc in documents:
        for q in walk(doc):
            try: blocked.add(task_key(q))
            except (ValueError, KeyError, TypeError, ZeroDivisionError): pass
    initial = len(blocked)
    rng = random.Random(570057)
    splits = {}
    for split, n in [('train',16), ('holdout',8)]:
        result = []
        for domain, targets, build in [('events',EVENTS,event), ('moments',MOMENTS,moment)]:
            for i in range(n):
                for attempt in range(10000):
                    den = (100, 101, 37, 13)[i%4]
                    params = (rng.randrange(1,den),rng.randrange(1,den),den,bool((i//4+i//2)%2)) if domain=='events' else (
                        rng.randrange(10,100),rng.randrange(10,651),rng.randrange(2,10),rng.randrange(1,31))
                    if domain=='events' and (params[0]==params[1] or params[0]+params[1]==den):continue
                    qs = [build(params,t,split,i,view) for t in targets for view in (0,1)]
                    keys = {task_key(q) for q in qs}
                    if len(keys)==len(targets) and not keys & blocked:break
                else: raise RuntimeError('Could not find disjoint inputs')
                blocked.update(keys);result.extend(qs)
        splits[split] = result
    assert len(splits['train'])==256 and len(splits['holdout'])==128
    supplements = []
    for q in splits['train']:
        supplements.append(dict(source_id=q['id'],story_id=q['story_id'],category=q['category'],
            prompt=q['question']+SUFFIX,target='Expression: '+q['expression'], semantic_key=task_key(q),
            source_group='semantic_supplement',view=q['view'],
            provenance='assistant_authored_domain_oracle_verified'))
    pool = defaultdict(list)
    for row in base:
        k = row['semantic_key']
        if k[0] in ('events','moments'): pool[(k[0],k[1])].append(row)
    for items in pool.values(): rng.shuffle(items)
    repeat, counts = [], Counter()
    for new in supplements:
        k = tuple(new['semantic_key'][:2]); candidates = pool[k]
        row = copy.deepcopy(candidates[counts[k] % len(candidates)])
        counts[k] += 1;row['repeat_source_id']=row['source_id']
        row.update(source_group='repeat_supplement', source_id='repeat_'+new['source_id'],
                   story_id=new['story_id'])
        repeat.append(row)
    # Use the same index permutation for both arms. Shared rows have identical
    # positions; every changed supplement slot has the same semantic target.
    permutation = list(range(1006));random.Random(2027).shuffle(permutation)
    arms = {arm:[(base+extra)[i] for i in permutation]
            for arm,extra in [('repeat_control',repeat),('semantic_course',supplements)]}
    assert all(len(rows)==1006 for rows in arms.values())
    for left,right in zip(*arms.values()):
        assert tuple(left['semantic_key'][:2])==tuple(right['semantic_key'][:2])
    test_keys = {task_key(q) for q in splits['holdout']}
    for rows in arms.values(): assert not {frozen(r['semantic_key']) for r in rows} & test_keys
    output.parent.mkdir(parents=True,exist_ok=True)
    data = dict(arms=arms,train_questions=splits['train'],holdout=splits['holdout'],
        manifest=dict(status='frozen_before_training',historical_blocked_identities=initial,
            corrected_source_sha256=digest(base),arm_sha256={k:digest(v) for k,v in arms.items()},
            holdout_sha256=digest(splits['holdout']),new_train_identities=128,new_holdout_identities=64,
            shared_rows=750,supplement_rows=256,updates_per_arm=126,teacher_calls=0))
    output.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data['manifest'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--corrected',type=Path,default=ROOT/'docs/STATS_TEACHER_CORRECTED_CORPUS.json')
    p.add_argument('--output',type=Path,default=ROOT/'docs/STATS_SEMANTIC_COURSE_DATA.json')
    a=p.parse_args();generate(a.archive,a.corrected,a.output)
