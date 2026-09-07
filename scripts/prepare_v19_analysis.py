"""Fixed-weight diagnostic corpus and permanent anchor specification."""
import copy,json,hashlib
from pathlib import Path
from stats_curriculum_v0_18 import make,TRACKS,prompt
from stats_curriculum_v0_19 import score
from stats_curriculum_v0_13 import digest
from exact_calculator import calculate
from stats_holdout_v1 import questions
from run_stats_rotation_v1 import rotate
from stats_v0_3_common import prompt_for
D=Path(__file__).resolve().parents[1]/'docs'
x=json.loads((D/'STATS_V0_19_FROZEN_QUESTIONS.json').read_text())
def paraphrase(q):
 a,m,c,b,v,u,k=q['parameters'];t=q['track'];d=q['depth'];s=m*60+30
 texts={
 'poisson_variance':{1:f'The expected value of a Poisson-distributed count X is {a}. Report the variance of this count.',2:f'Consider a {m}-minute observation. Events follow a homogeneous Poisson process at {a} per minute. Report the variance of the number observed.',3:f'The observation lasts {s} seconds. Events follow a homogeneous Poisson process at {a} per minute. Report the variance of the number observed.'},
 'scaled_variance':{1:f'Report the variance of Y. Define Y by multiplying X by {c} and then adding {b}; the variance of X is {v}.',2:f'Report the variance of Y. Define Y by multiplying X by {c} and then adding {b}; X is a Poisson count whose expectation is {a}.',3:f'Report the variance of Y, where Y={c}*X+{b}. X counts events during {m} minutes; these events follow a homogeneous Poisson process at {a} per minute.'},
 'second_moment':{1:f'Report the expected value of the square of X. Its variance is {v}, and its mean is {a}.',2:f'Report the expected value of the square of X. X follows a Poisson distribution with expected value {a}.',3:f'Report the expected value of X squared. X counts events over {m} minutes, from a homogeneous Poisson process at {a} events per minute.'},
 'conditional_wait':{1:f'Report the average total duration T, in minutes. T has a continuous uniform distribution with lower bound {k} minutes and upper bound {u} minutes.',2:f'You know T has already exceeded {k} minutes. Before this information, T was uniform between 0 and {u} minutes. Report the conditional expected total T, in minutes, not the remaining time.',3:f'You know T has already exceeded {k*60+30} seconds. Before this information, T was uniform between 0 and {u} minutes. Report the conditional expected total T, in minutes, not the remaining time.'}}
 return texts[t][d]
rows=[]
for t in TRACKS:
 for depth in (1,2,3):
  selected=sorted([q for q in x['new_test'] if q['track']==t and q['depth']==depth],key=lambda q:q['id'])[:2]
  for q in selected:
   for variant in ('original','paraphrase','magnitude'):
    r=copy.deepcopy(q)
    if variant=='paraphrase':r['question']=paraphrase(q)
    if variant=='magnitude':
     p=list(q['parameters']);p[0]*=10;p[1]*=10;p[4]*=100;p[5]*=10;p[6]*=10
     r=make(t,depth,p,'diagnostic',len(rows))
    r.update(id='analysis_'+q['id']+'_'+variant,variant=variant,pair_id=q['id'],story_id=q['story_id'],probe='concept')
    rows.append(r)
for q in sorted([q for q in x['old_test'] if q['category']=='exactly_one'],key=lambda q:q['id'])[:8]:
 a,b=q['bindings']['miss_a'],q['bindings']['miss_b']
 for kind,word in [('exactly_one','exactly one'),('at_least_one','at least one')]:
  r=copy.deepcopy(q);expr=f'(1-{a})*({b})+({a})*(1-{b})' if kind=='exactly_one' else f'1-({a})*({b})'
  r.update(id='analysis_'+q['id']+'_'+kind,category=kind,question=f'Two independent methods inspect the same defect. The probability that A misses it is {a}; the probability that B misses it is {b}. Find the probability that {word} method detects the defect.',expression=expr,answer=calculate(expr),variant=kind,probe='event_contrast',pair_id=q['id'],story_id=q['id']);rows.append(r)
assert len(rows)==88 and len({q['id'] for q in rows})==88
for q in rows:assert score('Expression: '+q['expression'],q)['correct'],q['id']
mc=questions();rotations=[dict(id=q['id'],shift=s,prompt=prompt_for(rotate(q,s)),expected=rotate(q,s)['answer_letter']) for q in mc for s in range(4)]
source=Path(__file__).resolve().parent
names=['stats_holdout_v1.py','run_stats_v0_4.py','run_stats_rotation_v1.py','stats_v0_3_common.py','stats_curriculum_v0_19.py','stats_curriculum_v0_18.py','formulation_grader.py','exact_calculator.py']
anchors=dict(adopted_after_v19=True,exposed_not_blind=True,mc=dict(questions=mc,rotations=rotations,n_questions=60,n_responses=240,group_unit='original question',max_new_tokens=16,do_sample=False),formulation=dict(questions={k:x[k] for k in ('old_test','new_test')},prompts={q['id']:prompt(q) for k in ('old_test','new_test') for q in x[k]},max_new_tokens=160,do_sample=False),source_hashes={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in names},policy='Never overwrite this anchor. Any prompt, grader, reference or runtime change creates a separately named series. Preserve raw outputs. Semantic credits versioned separately. Unknown historical compatibility is a gap, not an interpolated curve.')
(D/'STATS_PERMANENT_ANCHOR_V1.json').write_text(json.dumps(anchors,indent=2)+'\n')
(D/'STATS_V19_PERTURBATION_QUESTIONS.json').write_text(json.dumps(rows,indent=2)+'\n')
print('PROBE',digest(rows),'ANCHOR',digest(anchors))
