"""Meaning-preserving prompt augmentation with unchanged verified teacher targets.

Numerical references are used for checks and evaluation, never as new training
answers. Style 0 is training-only; style 1 is reserved for diagnostic evaluation.
"""
import argparse
import copy
import hashlib
import json
import random
import re
from fractions import Fraction as F
from pathlib import Path

from exact_calculator import calculate
from stats_consolidation_semantics import spec_for, oracle, task_key, validate_question
from stats_consolidation_pilot import build, digest
from stats_curriculum_v0_13 import KINDS, make

ROOT = Path(__file__).resolve().parents[1]
SUFFIX = ('\nReturn one line: Expression: <a fully substituted numerical expression>. '
          'Keep arithmetic and unit conversion operations unevaluated. Use integers, '
          'fractions, +, -, *, /, ** and comb(n,r). No final answer or explanation.')


def read(p):
    return json.loads(Path(p).read_text())


def number(x):
    x = F(x)
    return str(x.numerator) if x.denominator == 1 else f'{x.numerator}/{x.denominator}'


def duration(x):
    # Preserve the supplied seconds-to-minutes task, including unevaluated /60.
    return (f'{number(F(x)*60)} seconds' if '/60' in str(x).replace(' ', '')
            else f'{number(x)} minutes')


def render(q, style):
    assert style in (0, 1)
    s = spec_for(q)
    k = s['kind']
    if k == 'interval':
        lo, hi, factor = number(s['lower']), number(s['upper']), number(F(s['divisor'])**2)
        if style == 0:
            return (f'The left and right confidence limits were {lo} and {hi}. '
                    f'A new sample has {factor} times as many observations. The estimate at the center, '
                    'confidence level, and population standard deviation stay fixed. '
                    'For this normal-theory interval, determine the updated right confidence limit.')
        return (f'Find the upper limit of a recomputed normal-theory confidence interval. '
                f'The previous limits were {lo} and {hi}; the sample now contains {factor} times '
                'the previous number of observations. Keep the center, confidence level, '
                'and population standard deviation at their previous values.')
    if k == 'uniform':
        lo, hi = number(s['lower']), number(s['upper'])
        if not s['conditional']:
            return ([f'A duration in minutes is uniformly distributed between {lo} and {hi}. '
                     'What is its expected duration?',
                     f'Give the average duration in minutes for a uniform distribution with '
                     f'lower boundary {lo} and upper boundary {hi}.'][style])
        cut = duration(s['cutoff'])
        return ([f'The full duration T, measured in minutes, is uniform between {lo} and {hi}. '
                 f'We learn that T is longer than {cut}. Find the expected full duration in '
                 'minutes under that condition, including the time already elapsed.',
                 f'A clock measures an entire wait T, uniform from {lo} to {hi} minutes. '
                 f'Condition on the clock eventually showing more than {cut}. '
                 'What is the expected entire wait in minutes in this conditional distribution?'][style])
    if k == 'binomial':
        n, r, prob = s['n'], s['r'], number(F(s['p'])*100)
        return ([f'There are {n} independent decisions, all under true null hypotheses. '
                 f'Each has a {prob}% rejection probability. Express the chance of recording '
                 f'exactly {r} rejections.',
                 f'Among {n} mutually independent null-hypothesis tests, exactly {r} are to '
                 f'reject. Each null is true and each rejection probability is {prob}%. '
                 'What is the probability of that exact count?'][style])
    if k == 'events':
        miss = 'miss_a' in q.get('bindings', {})
        pa, pb = F(s['p']), F(s['p_b'])
        label = 'miss' if miss else 'detect'
        pa, pb = (1-pa, 1-pb) if miss else (pa, pb)
        def probability(x):
            return number(x*100)+'%' if (x*100).denominator==1 else number(x)
        a, b = probability(pa), probability(pb)
        targets = {
            'exactly_one': ('one detector registers the defect and the other does not', 'the two detection outcomes differ'),
            'at_least_one': ('one or two detectors register the defect', 'there is a detection from either or both detectors'),
            'both': ('both detectors register the defect', 'each detector reports a detection'),
            'neither': ('neither detector registers the defect', 'both detectors fail to report a detection'),
            'same': ('the detectors give matching detection outcomes', 'the detection outcomes are equal, whether positive or negative')}
        event = targets[s['target']][style]
        return ([f'A present defect is checked independently by detectors A and B. '
                 f'The probabilities that they {label} it are {a} and {b}, respectively. '
                 f'Calculate the probability that {event}.',
                 f'For a defect that is present, A and B make independent detection decisions. '
                 f'A has probability {a} to {label} it and B has probability {b} to {label} it. '
                 f'We want the probability that {event}.'][style])
    if k == 'process':
        rate, window = number(s['rate']), duration(s['duration'])
        facts = [f'X records the number of arrivals during {window}. Arrivals follow a '
                 f'homogeneous Poisson process with an average of {rate} per minute.',
                 f'Observe a homogeneous Poisson arrival process for {window} and call the '
                 f'resulting count X. Its arrival intensity is {rate} per minute.'][style]
    elif k == 'poisson':
        facts = [f'The count X has a Poisson distribution with expected count {number(s["mean"])}.',
                 f'Consider a Poisson-distributed variable X whose expectation is {number(s["mean"])}.'][style]
    elif k == 'moments':
        if s['target'] == 'variance':
            facts = f'The variance of a random variable X is {number(s["variance"])}.'
        else:
            facts = [f'A random variable X has expectation {number(s["mean"])} and variance {number(s["variance"])}.',
                     f'For X, the variance is {number(s["variance"])} and the expected value is {number(s["mean"])}.'][style]
    else:
        raise ValueError(k)
    var = 'X'
    if 'scale' in s:
        var = 'Y'
        a, b = number(s['scale']), number(s.get('offset', 0))
        facts += [f' A second recorded variable is Y={a}*X+{b}.',
                  f' Define Y by multiplying X by {a} and then adding {b}.'][style]
    target = {'variance': [f'What is the variance of {var}?', f'Give Var({var}).'],
              'mean': [f'What is the expected value of {var}?', f'Give E[{var}].'],
              'second_moment': [f'What is the expected value of {var} squared?',
                                f'Give the second raw moment E[{var}**2].']}[s['target']][style]
    # "expected value of X squared" can mean (E[X])^2: make the scope explicit.
    if s['target'] == 'second_moment' and style == 0:
        target = f'Square {var} first, then take its expectation. What is E[{var}**2]?'
    return facts+' '+target


def variant(q, style, ident):
    out = copy.deepcopy(q)
    out.update(id=ident, question=render(q, style), semantics=spec_for(q))
    validate_question(out)
    assert task_key(out) == task_key(q)
    return out


def replace_numbers(template_question, old, new):
    """Change numeric parameters while preserving the exact known prose."""
    def values(q):
        s=spec_for(q)
        if q['category']=='poisson_time':return [number(s['rate']),number(F(s['duration'])*60)]
        if q['category']=='uniform_time':return [number(s['upper']),number(F(s['cutoff'])*60)]
        return [number(s['lower']),number(s['upper']),number(F(s['divisor'])**2)]
    a,b=values(old),values(new)
    assert len(set(a))==len(a)
    pattern=r'(?<!\w)\d+(?:\.\d+)?(?!\w)'
    seen=re.findall(pattern,template_question)
    assert set(a)<=set(seen), (a,template_question)
    mapping=dict(zip(a,b))
    return re.sub(pattern,lambda m:mapping.get(m[0],m[0]),template_question)


def prepare(verified_path, output):
    verified = read(verified_path)
    historical = {q['id']: q for v in (14, 15)
                  for q in read(ROOT/f'docs/STATS_V0_{v}_FROZEN_QUESTIONS.json')['train']}
    candidates = {q['id']: q for story in build()[0] for q in story['questions']}
    replay = [r for r in read(ROOT/'docs/STATS_V0_19_REPLAY_SOURCE.json') if r['category'] in KINDS]
    assert len(replay) == 192 and len(verified['train']) == 67
    rows, source_questions = [], []
    for r in replay+verified['train']:
        q = historical.get(r['source_id']) or candidates[r['source_id']]
        validate_question(q)
        expression = r['target'].removeprefix('Expression: ').strip()
        assert F(calculate(expression)) == oracle(spec_for(q))
        assert r['prompt'].split('\n')[0] == q['question']
        rewritten = variant(q, 0, 'multiview_'+q['id'])
        source_questions.append(q)
        for view, prompt in [('original', r['prompt']), ('paraphrase', rewritten['question']+SUFFIX)]:
            rows.append(dict(source_id=r['source_id'], story_id=r.get('story_id',r['source_id']),
                category=r['category'], source_group='replay' if r in replay else 'new', view=view,
                prompt=prompt, target=r['target'], semantic_key=task_key(q),
                source_row_sha256=digest(r), original_question_sha256=digest(q),
                target_sha256=hashlib.sha256(r['target'].encode()).hexdigest(),
                provenance='unchanged_verified_70B_target_with_semantics_preserving_prompt_augmentation'))
    assert len(rows) == 518
    unique, duplicates = {}, []
    for row in rows:
        if row['prompt'] in unique:
            prior=unique[row['prompt']]
            assert prior['semantic_key']==row['semantic_key']
            duplicates.append({'kept_source_id':prior['source_id'],'removed_source_id':row['source_id'],
                               'same_target':prior['target']==row['target'],'view':row['view']})
        else:unique[row['prompt']]=row
    rows=list(unique.values())
    # Block known train/validation/test identities; diagnostic new parameters
    # must not reuse any existing mathematical task, even under different wording.
    blocked = {task_key(q) for q in source_questions}
    blocked.update(task_key(q) for q in candidates.values())
    for group in read(ROOT/'docs/STATS_CONSOLIDATION_HOLDOUT.json')['suites'].values():
        blocked.update(task_key(q) for q in group)
    for p in (ROOT/'docs').glob('STATS_V0_*_FROZEN_QUESTIONS.json'):
        for group in read(p).values():
            if not isinstance(group,list): continue
            for q in group:
                try: blocked.add(task_key(q))
                except (ValueError, KeyError, TypeError): pass
    matrix = read(ROOT/'docs/STATS_CONSOLIDATION_SELECTION_VALIDATION.json')
    for group in matrix['suites'].values():
        blocked.update(task_key(q) for q in group)
    assert not ({r['prompt'] for r in rows} & {q['question']+SUFFIX for g in matrix['suites'].values() for q in g})
    probe = []
    rng = random.Random(550055)
    for category in ('poisson_time', 'uniform_time', 'interval'):
        known = [q for q in source_questions[:192] if q['category']==category][:4]
        for i, old in enumerate(known):
            while True:
                params = (rng.randrange(23,94),rng.randrange(61,571),rng.randrange(2,9),rng.randrange(2,19))
                fresh = make(category, params, 'multiview_probe', i)
                if task_key(fresh) not in blocked:
                    blocked.add(task_key(fresh));break
            for numbers, q in [('known',old),('new',fresh)]:
                for wording in ('familiar','unseen'):
                    item = copy.deepcopy(q) if wording=='familiar' else variant(q,1,q['id'])
                    # Exact source wording for known numbers, and ONLY numeric
                    # substitutions for new numbers, isolate the numbers axis.
                    if wording=='familiar':
                        item['question']=replace_numbers(old['question'],old,q)
                    item.update(id=f'multiview_probe_{category}_{i}_{numbers}_{wording}',
                                numbers=numbers,wording=wording,pair_id=f'{category}_{i}',semantics=spec_for(q))
                    validate_question(item)
                    probe.append(item)
    assert len(probe)==48
    artifact={'train':rows,'transfer_probe':probe,'manifest':{
        'teacher_calls':0,'verified_source_rows':259,'training_rows':len(rows),'duplicates_removed':duplicates,
        'answer_source':'unchanged previously verified teacher responses',
        'styles':{'train':'original plus style 0','probe':'fixed familiar plus unseen style 1'},
        'probe_scope':'3 families x 4 paired stories x 2 number groups x 2 wording groups; known parameters are fitting diagnostics, not holdout',
        'verified_teacher_file_sha256':hashlib.sha256(Path(verified_path).read_bytes()).hexdigest()}}
    artifact['manifest']['train_sha256']=digest(rows)
    artifact['manifest']['probe_sha256']=digest(probe)
    output.write_text(json.dumps(artifact,indent=2)+'\n')
    print(json.dumps(artifact['manifest'],indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--verified',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.verified,a.output)
