"""Frozen, one-attempt teacher diagnostic; never used as student training data."""
import copy
import json
from pathlib import Path
from stats_curriculum_v0_13 import KINDS, make, digest, prompt
from stats_curriculum_v0_14 import reword
from formulation_grader import grade
from flight_run_stats_v0_3 import TeacherClient, save_json, package

ROOT=Path('/kaggle/working/3beethoven_teacher_scaffold_v0_14')


def build():
    pairs=[]
    def add(group,a,b):
        i=len(pairs)
        for j,q in enumerate((a,b)):q['id']=f'teacher_scaffold_{i:02d}_{j}'
        pairs.append(dict(id=i,group=group,questions=[a,b]))
    for i,kind in enumerate(KINDS):
        p=[71+i,613+7*i,6,23]
        a=make(kind,p,'teacher_check',i)
        b=copy.deepcopy(a);b['question']=reword(b,True)
        add('wording',a,b)
        a=make(kind,p,'teacher_check',i)
        changed=p.copy();changed[0]+=7
        b=make(kind,changed,'teacher_check',i)
        add('parameter',a,b)
    for i in range(4):
        rate,minutes=83+i,11+i
        a=make('poisson_time',[rate,minutes*60,5,19],'teacher_check',i)
        b=copy.deepcopy(a)
        b['question']=f'A homogeneous Poisson counter averages {rate} events per minute. Find the count variance over {minutes} minutes.'
        b['expression']=f'{rate}*{minutes}'
        b['bindings']['duration_minutes']=str(minutes)
        add('unit',a,b)
    for i in range(4):
        p=[81+i,641+7*i,6,23]
        add('event',make('exactly_one',p,'teacher_check',i),make('at_least_one',p,'teacher_check',i))
    return dict(formula_reminders='General formula reminders: Var(a*X+b)=a**2*Var(X); E[(a*X+b)**2]=a**2*Var(X)+(a*E[X]+b)**2. For a normal interval [L,U], multiplying sample size by k**2 gives new upper endpoint (L+U)/2+(U-L)/(2*k). Distinguish variance from second moment. Do not evaluate midpoint, powers, or other arithmetic. These are general rules, not the answer to the question.',pairs=pairs,gate=dict(min_verified=44,min_pairs=20,min_group_fraction=.75),
                policy='48 independent prompts, one response each, no retries. Pending review does not pass. General formula reminders supplied; no question-specific reference answers sent. Freeze before calls. No training on this diagnostic.')


def main():
    from kaggle_secrets import UserSecretsClient
    frozen=build();ROOT.mkdir(exist_ok=True)
    committed=json.loads((Path(__file__).resolve().parents[1]/'docs/STATS_V0_14_TEACHER_SCAFFOLD.json').read_text())
    assert committed==frozen
    save_json(ROOT/'protocol.json',dict(**frozen,sha256=digest(frozen)))
    client=TeacherClient(ROOT,UserSecretsClient().get_secret('OPENROUTER_API_KEY'),48)
    rows=[]
    try:
        for pair in frozen['pairs']:
            for q in pair['questions']:
                usage=client.stats()
                if usage['reported_cost_usd']>=.20 or usage['responses_without_cost']:
                    raise RuntimeError('Cost reporting gate')
                raw=client.call(q['id'],[dict(role='system',content=frozen['formula_reminders']+' Solve independently. Return a fully substituted numerical expression, not its evaluated final answer. Keep operations and conversion factors unevaluated. No explanation.'),dict(role='user',content=prompt(q))],max_tokens=200)
                rows.append(dict(pair_id=pair['id'],group=pair['group'],id=q['id'],raw=raw,judged=grade(raw,q)))
                save_json(ROOT/'responses.json',rows)
            print('SCAFFOLD',len(rows),'/48',flush=True)
        count=lambda rs:sum(r['judged']['math_correct'] is True for r in rs)
        groups={g:dict(n=len(rs:=[r for r in rows if r['group']==g]),verified=count(rs)) for g in ('wording','parameter','unit','event')}
        pairs=sum(all(r['judged']['math_correct'] is True for r in rows if r['pair_id']==p['id']) for p in frozen['pairs'])
        passed=count(rows)>=44 and pairs>=20 and all(v['verified']/v['n']>=.75 for v in groups.values())
        summary=dict(n=len(rows),verified=count(rows),pairs_verified=pairs,pairs_total=24,groups=groups,passed=passed,usage=client.stats(),protocol_sha256=digest(frozen))
        save_json(ROOT/'summary.json',summary)
        print('SCAFFOLD_RESULT',json.dumps(summary),flush=True)
    finally:package(ROOT)


if __name__=='__main__':main()
