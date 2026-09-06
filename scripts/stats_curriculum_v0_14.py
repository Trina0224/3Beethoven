"""Frozen wording-transfer pilot. No teacher-generated test questions."""
import json
import random
from pathlib import Path
from stats_curriculum_v0_13 import KINDS, make, digest, prompt, score


def reword(q,test=False):
    b=q['bindings']; c=q['category']
    if c=='poisson_time':
        s=b['duration_minutes'].split('/')[0]; r=b['rate_per_minute']
        return (f'The arrivals at a counter follow a homogeneous Poisson process, averaging {r} per minute. A recording lasts {s} seconds. What is the variance of the number recorded?' if not test else
                f'Count all arrivals in a {s}-second observation window. The count comes from a homogeneous Poisson arrival process whose mean rate is {r} per minute. Set up the count variance.')
    if c=='poisson_scaled':
        m,a,z=b['mean'],b['scale'],b['offset']
        return (f'A sensor first samples X from a Poisson distribution with mean {m}. It multiplies X by {a}, then adds {z}, producing Y. Set up the variance of Y.' if not test else
                f'Adding {z} after multiplying a Poisson count X by {a} gives the displayed value. X has expectation {m}. What variance does this display have?')
    if c=='moment':
        m,v,a,z=b['mean'],b['variance'],b['scale'],b['offset']
        return (f'X has mean {m} and variance {v}. Multiply each observation by {a} and add {z} to get Y. Find the expected square of Y.' if not test else
                f'The quantity to average is the square of ({a} times X plus {z}), not the square of its average. The mean of X is {m}, and its variance is {v}. Set up that expectation.')
    if c=='uniform_time':
        u,s=b['upper_minutes'],b['cutoff_minutes'].split('/')[0]
        return (f'A wait is uniformly distributed between zero and {u} minutes. We learn the total wait is longer than {s} seconds. Set up the conditional expected TOTAL wait in minutes, not the remaining wait.' if not test else
                f'Only waits exceeding {s} seconds are retained. Before this selection, total waiting time is uniform on [0,{u}] minutes. What is the mean total duration, in minutes, of retained waits?')
    if c=='binomial':
        n,r,p=b['n'],b['r'],b['reject_probability'].split('/')[0]
        return (f'All {n} null hypotheses are true. Their tests act independently and each falsely rejects in {p} percent of cases. Set up the chance of precisely {r} false rejections.' if not test else
                f'An experiment repeats {n} independent tests under true null hypotheses. A rejection occurs with probability {p}/100 on each test. We count rejections and want the probability that the count equals {r}.')
    if c in ('exactly_one','at_least_one'):
        a,beta=b['miss_a'].split('/')[0],b['miss_b'].split('/')[0]
        event=('one method detects and the other misses' if c=='exactly_one' else 'one or both methods detect')
        return (f'A defect is definitely present. Two independent inspections A and B miss it in {a}% and {beta}% of cases respectively. Find the probability that {event}.' if not test else
                f'For an existing defect, inspection A has miss probability {a}/100, and independent inspection B has miss probability {beta}/100. What is the chance that {event}?')
    lo,hi,k=b['lower'],b['upper'],b['width_divisor']; factor=int(k)**2
    return (f'The endpoints of a normal-theory interval are {lo} and {hi}. Its center, population standard deviation and confidence level will not change. A new sample has {factor} times the old size. Set up the new upper endpoint.' if not test else
            f'Increase sample size by a factor of {factor}, leaving confidence level, population standard deviation and interval center unchanged. The original normal-theory interval runs from {lo} to {hi}. Where is its upper endpoint after this change?')


def build():
    docs=Path(__file__).resolve().parents[1]/'docs'
    old=json.loads((docs/'STATS_V0_13_FROZEN_QUESTIONS.json').read_text())
    blocked={tuple(q['identity']) for rows in old.values() for q in rows}
    # Also exclude old numerical answers, avoiding identity-alias leakage and
    # any exact probability cases present in earlier saved curricula.
    answers={q['answer'] for rows in old.values() for q in rows}
    for version in (9,10,11,12):
        prev=json.loads((docs/f'STATS_V0_{version}_FROZEN_QUESTIONS.json').read_text())
        answers.update(q['answer'] for rows in prev.values() for q in rows)
    rng=random.Random(1414);data={s:[] for s in ('train','validation','test')}
    for split,count in (('train',16),('validation',4),('test',8)):
        for kind in KINDS:
            added=0
            for _ in range(100000):
                p=(rng.randrange(21,49),rng.randrange(61,599),rng.randrange(2,8),rng.randrange(2,19))
                q=make(kind,p,split,added);key=tuple(q['identity'])
                if key in blocked or q['answer'] in answers:continue
                blocked.add(key);answers.add(q['answer'])
                q['id']=q['id'].replace('v13_','v14_');q['question']=reword(q,split=='test')
                q['wording_split']='heldout_wording' if split=='test' else 'instruction_wording'
                data[split].append(q);added+=1
                if added==count:break
            assert added==count,(split,kind,added)
    assert {s:len(v) for s,v in data.items()}==dict(train=128,validation=32,test=64)
    return data


if __name__=='__main__':
    d=build();p=Path(__file__).resolve().parents[1]/'docs/STATS_V0_14_FROZEN_QUESTIONS.json'
    p.write_text(json.dumps(d,indent=2)+'\n');print(digest(d))
