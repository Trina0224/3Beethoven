"""Frozen direct-interface final experiment data; no teacher calls."""
import argparse,hashlib,json,random,zipfile
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
from stats_consolidation_semantics import oracle,task_key,validate_question
from stats_consolidation_grader import score
from exact_calculator import calculate
ROOT=Path(__file__).resolve().parents[1]
SUFFIX='\nReturn one line: Expression: <a fully substituted numerical expression>. Keep arithmetic and unit conversion operations unevaluated. Use integers, fractions, +, -, *, /, ** and comb(n,r). No final answer or explanation.'
CATS=['both','neither','exactly_one','at_least_one','same','moment_mean','moment_variance','moment_second','poisson_variance','poisson_scaled','poisson_second','process_variance','process_scaled','process_second','uniform_mean','uniform_conditional','binomial','interval']
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def walk(x):
 if isinstance(x,dict):
  yield x
  for y in x.values():yield from walk(y)
 elif isinstance(x,list):
  for y in x:yield from walk(y)
def draw(cat,rng,i):
 m=rng.randrange(11,100);v=rng.randrange(21,651);a=rng.randrange(2,10);b=rng.randrange(2,31)
 if cat in CATS[:5]:
  den=[100,101,37,13][i%4];p=rng.randrange(1,den);q=rng.randrange(1,den)
  s=dict(kind='events',target=cat,p=f'{p}/{den}',p_b=f'{q}/{den}')
  x=f'({p}/{den})';y=f'({q}/{den})';e={'both':f'{x}*{y}','neither':f'(1-{x})*(1-{y})','exactly_one':f'{x}*(1-{y})+(1-{x})*{y}','at_least_one':f'1-(1-{x})*(1-{y})','same':f'{x}*{y}+(1-{x})*(1-{y})'}[cat]
  ask={'both':'both occur','neither':'neither occurs','exactly_one':'exactly one occurs','at_least_one':'at least one occurs','same':'both occur or neither occurs'}[cat]
  text=f'A and B are independent events. P(A)={p}/{den}; P(B)={q}/{den}. Find the probability that {ask}.'
 elif cat.startswith('moment'):
  target={'moment_mean':'mean','moment_variance':'variance','moment_second':'second_moment'}[cat]
  s=dict(kind='moments',target=target,mean=str(m),variance=str(v),scale=str(a),offset=str(b))
  e={'mean':f'{a}*{m}+{b}','variance':f'{a}**2*{v}','second_moment':f'{a}**2*{v}+({a}*{m}+{b})**2'}[target]
  ask={'mean':'E[Y]','variance':'Var(Y)','second_moment':'E[Y**2]'}[target]
  text=f'E[X]={m}; Var(X)={v}. Define Y={a}*X+{b}. Find {ask}.'
 elif cat.startswith(('poisson','process')):
  process=cat.startswith('process');scaled=cat.endswith('scaled');target='second_moment' if cat.endswith('second') else 'variance'
  s=dict(kind='process' if process else 'poisson',target=target)
  if process:
   if i%2:t=rng.randrange(2,16);dt=str(t);unit='minutes'
   else:t=rng.randrange(61,901);dt=f'{t}/60';unit='seconds'
   s.update(rate=str(m),duration=dt);lam=f'({m}*({dt}))'
   text=f'A homogeneous Poisson process has rate {m} arrivals per minute. X counts arrivals during {t} {unit}. '
  else:
   if cat in ('poisson_variance','poisson_second'):m=rng.randrange(11,501)
   s['mean']=str(m);lam=str(m);text=f'X has a Poisson distribution with mean {m}. '
  if scaled:
   s.update(scale=str(a),offset=str(b));e=f'{a}**2*{lam}';text+=f'Define Y={a}*X+{b}. Find Var(Y).'
  elif target=='variance':e=lam;text+='Find Var(X).'
  else:e=f'{lam}+{lam}**2';text+='Find E[X**2].'
 elif cat.startswith('uniform'):
  u=rng.randrange(22,91);lo=rng.randrange(1,u-2)
  if cat=='uniform_mean':
   s=dict(kind='uniform',conditional=False,lower=str(lo),cutoff=str(lo),upper=str(u));e=f'({lo}+{u})/2';text=f'T is uniform on [{lo}, {u}] minutes. Find E[T] in minutes.'
  else:
   cut=lo*60+rng.randrange(1,60);s=dict(kind='uniform',conditional=True,lower='0',cutoff=f'{cut}/60',upper=str(u));e=f'({cut}/60+{u})/2';text=f'T is uniform on [0, {u}] minutes. Given T>{cut} seconds, find the conditional mean of TOTAL T in minutes.'
 elif cat=='binomial':
  n=rng.randrange(5,31);r=rng.randrange(2,n-1);p=rng.randrange(3,90)
  s=dict(kind='binomial',n=str(n),r=str(r),p=f'{p}/100');e=f'comb({n},{r})*({p}/100)**{r}*(1-{p}/100)**({n}-{r})';text=f'X is the number of successes in {n} independent trials, each with success probability {p}/100. Find P(X={r}).'
 else:
  lo=rng.randrange(11,100);hi=lo+rng.randrange(21,151);d=rng.randrange(2,8)
  s=dict(kind='interval',lower=str(lo),upper=str(hi),divisor=str(d));e=f'({lo}+{hi})/2+({hi}-{lo})/(2*{d})';text=f'A normal-theory confidence interval is [{lo}, {hi}]. Sample size is multiplied by {d*d}; center, confidence level and population standard deviation stay fixed. Find the new upper endpoint.'
 bindings=dict(lower=s['lower'],upper=s['upper'],width_divisor=s['divisor']) if cat=='interval' else {}
 return dict(category=cat,question=text,semantics=s,expression=e,answer=str(oracle(s)),bindings=bindings)
def source_match(q,r):
 k=r['semantic_key'];s=q['semantics']
 if k[0]!=s['kind']:return False
 if k[0] in ('events','moments','poisson','process'):return k[1]==s['target']
 if k[0]=='uniform':return k[1]==s['conditional']
 return True
def build(archive):
 z=zipfile.ZipFile(archive);source=json.loads(z.read('source/docs/STATS_SEMANTIC_COURSE_DATA.json'))['arms']['repeat_control']
 blocked=set()
 def take(obj):
  for x in walk(obj):
   try:
    if 'semantic_key' in x:blocked.add(json.dumps(x['semantic_key'],sort_keys=True))
    if 'question' in x and 'category' in x and ('semantics' in x or 'bindings' in x):blocked.add(json.dumps(task_key(x),sort_keys=True))
   except (ValueError,KeyError,TypeError):pass
 take(source)
 for p in (ROOT/'docs').glob('*.json'):
  if p.name.startswith('STATS_FINAL_CLEAR'):continue
  try:take(json.loads(p.read_text()))
  except (ValueError,UnicodeError):pass
 for n in z.namelist():
  if n.startswith('source/docs/') and n.endswith('.json'):
   try:take(json.loads(z.read(n)))
   except ValueError:pass
 initial=len(blocked);rng=random.Random(580058);data={};seen_stories={};anchors={}
 for split,n in [('train',64),('development',4),('test',8)]:
  rows=[]
  for cat in CATS:
   for i in range(n):
    for attempt in range(100000):
     q=draw(cat,rng,i);q['id']=f'final_clear_{split}_{cat}_{i:03d}';key=json.dumps(task_key(q),sort_keys=True)
     s=q['semantics'];story=dict(s);story.pop('target',None);storykey=json.dumps(story,sort_keys=True)
     # Full semantic identities and parameter stories cannot cross any split.
     if key not in blocked and seen_stories.get(storykey,split)==split:break
    else:raise RuntimeError(('Could not create disjoint task',cat,split,i))
    blocked.add(key);seen_stories[storykey]=split;validate_question(q)
    assert F(calculate(q['expression']))==oracle(s)
    if cat not in anchors:
     anchor=next((r for r in source if source_match(q,r)),None)
     anchors[cat]=None if anchor is None else {k:anchor[k] for k in ['source_id','semantic_key','target','provenance']}
    q.update(story_id=digest(story),prompt=q['question']+SUFFIX,target='Expression: '+q['expression'],semantic_key=task_key(q),provenance='assistant_authored_parameter_augmentation_of_verified_rule' if anchors[cat] else 'assistant_authored_domain_rule_no_teacher_anchor')
    # Some references allow precomputed Poisson means; mathematical review is explicit.
    g=score(q['target'],q);assert g['primary_correct'],(cat,g)
    rows.append(q)
  if split=='train':random.Random(2027).shuffle(rows)
  data[split]=rows
 data['manifest']=dict(status='frozen_before_inference',historical_keys=initial,categories=CATS,anchors=anchors,split_sha256={s:digest(data[s]) for s in data},counts={s:len(data[s]) for s in data},authoring='No new cloud teacher responses; explicit assistant-authored expansion of independently verified rules')
 return data
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('output');a=p.parse_args();d=build(a.archive);Path(a.output).write_text(json.dumps(d,ensure_ascii=False,indent=2));print(json.dumps(d['manifest'],ensure_ascii=False))
