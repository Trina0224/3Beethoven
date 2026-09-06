"""Fresh story/parameter probes, frozen before lessons; references are exact."""
from fractions import Fraction as F
from collections import Counter
from stats_v0_3_common import digest,normalize_question
from stats_curriculum_v0_5 import TOPICS

def build():
    rows=[]
    for topic in TOPICS:
        for v in range(8):
            k=v+3
            if topic=='poisson':
                if v%2==0:
                    q=f'A library scanner registers arrivals as a Poisson process with mean rate {k} per minute. Let N count arrivals during 3 minutes. What is Var(N)?'
                    ans=3*k; wrong=[k,9*k,9*k*k]
                else:
                    q=f'A museum counter X is Poisson with mean {k}. A display shows S=2X+5, using the same count twice. What is Var(S)?'
                    ans=4*k; wrong=[2*k,4*k+5,k]
            elif topic=='expectation':
                q=f'A game score X has mean {k} and variance 2. A scoreboard reports Y=2X+1. What is E[Y^2]?'
                ans=8+(2*k+1)**2; wrong=[(2*k+1)**2,8,4*(k*k+2)+1]
            elif topic=='uniform':
                q=f'A shuttle waiting time T is continuously uniform from 0 to {4*k} minutes. Given T>{2*k}, what is its conditional mean?'
                ans=3*k; wrong=[2*k,4*k,k]
            elif topic=='type_i':
                p=F(v+1,20)
                q=f'Three independent laboratory alarms each falsely trigger with probability {p} when no fault exists. What is the probability exactly two alarms falsely trigger?'
                ans=3*p*p*(1-p); wrong=[p*p,3*p*p,p**3]
            elif topic=='type_ii':
                b=F(v+2,20)
                q=f'Two independent inspection methods each miss an existing defect with probability {b}. What is the probability exactly one method detects the defect?'
                ans=2*b*(1-b); wrong=[(1-b)**2,1-b*b,b*b]
            else:
                q=f'A survey reports a normal-theory mean interval [{10*k},{10*k+12}]. Keep its center, confidence level and population standard deviation fixed, but multiply sample size by 9. What is the new upper endpoint?'
                ans=10*k+8; wrong=[10*k+12,10*k+10,10*k+6]
            choices=list(map(str,wrong));pos=(len(rows)*3+1)%4;choices.insert(pos,str(ans))
            rows.append(dict(id=f'v06_{topic}_{v:02d}',category=topic,family=topic+'_story_transfer',
                question=q,choices=choices,answer_letter='ABCD'[pos]))
    return rows

def audit(excluded=()):
    rows=build()
    assert len(rows)==48
    assert Counter(r['answer_letter'] for r in rows)==dict.fromkeys('ABCD',12)
    assert len({normalize_question(r['question']) for r in rows})==48
    assert not {normalize_question(r['question']) for r in rows}&{normalize_question(r['question']) for r in excluded}
    for r in rows: assert len({F(x) for x in r['choices']})==4,r['id']
    return dict(n=48,sha256=digest(rows),scope='New stories and parameters; known mathematical principles, internally authored, not unseen-skill or external-blind evaluation')
