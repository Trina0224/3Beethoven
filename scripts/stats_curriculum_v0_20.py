"""Event contrast curriculum; deterministic labels, no new teacher responses."""
import json, random
from fractions import Fraction
from pathlib import Path
from stats_curriculum_v0_19 import score, prompt, digest
from stats_curriculum_v0_13 import KINDS
from exact_calculator import calculate
DOCS=Path(__file__).resolve().parents[1]/'docs'
EVENTS=('exactly_one','both','neither','same')
def question(a,b,event,style,split,i):
    p=f'({a}/101)';q=f'({b}/101)'
    expr={'exactly_one':f'{p}*(1-{q})+(1-{p})*{q}', 'both':f'{p}*{q}', 'neither':f'(1-{p})*(1-{q})', 'same':f'{p}*{q}+(1-{p})*(1-{q})'}[event]
    asks=[{'exactly_one':'exactly one activates','both':'both activate','neither':'neither activates','same':'both activate or neither activates'},
          {'exactly_one':'one activates and the other does not','both':'the two activate together','neither':'the two both remain inactive','same':'the two have matching activation states'},
          {'exactly_one':'the indicators disagree','both':'there are two active indicators','neither':'there are zero active indicators','same':'the indicators agree'}]
    stem=(f'Two independent switches A and B activate with probabilities {a}/101 and {b}/101, respectively. ' if style==0 else
          f'Switch B activates with probability {b}/101; switch A activates with probability {a}/101. Their activations are independent. ' if style==1 else
          f'Independent indicators record the occurrence of events A and B as 1, and nonoccurrence as 0. P(B)={b}/101 and P(A)={a}/101. ')
    text=stem+'Find the probability that '+asks[style][event]+'.'
    return dict(id=f'v20_{split}_{i:03d}_{event}_s{style}',category=event,story_id=f'v20_{split}_{i:03d}',parameters=[a,b,101],question=text,bindings={},expression=expr,answer=calculate(expr),target='Expression: '+expr,provenance='procedural_reference; zero teacher calls')
def build():
    rng=random.Random(2020);pairs=[(a,b) for a in range(5,96) for b in range(a+1,97) if a+b!=101];rng.shuffle(pairs)
    result={};offset=0
    for split,n,styles in [('train',24,(0,1)),('validation',8,(0,1)),('test',24,(2,))]:
        rows=[]
        for i,(a,b) in enumerate(pairs[offset:offset+n]):
            for style in styles:
                for event in EVENTS:rows.append(question(a,b,event,style,split,i))
        result[split]=rows;offset+=n
    replay=json.loads((DOCS/'STATS_V0_19_REPLAY_SOURCE.json').read_text());replay=[r for r in replay if r['category'] in KINDS];assert len(replay)==192
    result['replay']=replay
    return result
if __name__=='__main__':
    d=build();(DOCS/'STATS_V0_20_FROZEN_QUESTIONS.json').write_text(json.dumps(d,indent=2)+'\n');print(digest(d))
