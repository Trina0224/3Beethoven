"""One focused teaching attempt per rejected preparation item; preserve initial run."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from flight_run_stats_v0_3 import TeacherClient,read_json,save_json,package
from prepare_stats_v0_15 import ROOT,teacher_rows
from stats_curriculum_v0_15 import build,prompt,score,digest

FOCUSED={
 'poisson_time':'For a homogeneous Poisson process with rate r events per minute and observation duration s seconds, the count variance is r*(s/60). This is a count, not an affine scaling of a random variable. Neither the rate nor the duration is squared. Substitute r and s from the question.',
 'uniform_time':'A total wait T uniform on [0,U] minutes, conditioned on T exceeding c seconds, is uniform on [c/60,U]. The expected TOTAL wait is (c/60+U)/2. Do not halve U before averaging, do not add the cutoff twice, and do not compute remaining wait.',
 'interval':'For old endpoints L,U and a sample-size multiplier M, the center is (L+U)/2. The new half-width is (U-L)/(2*sqrt(M)). The new UPPER endpoint is CENTER PLUS NEW HALF-WIDTH. Do not add the half-width to the old lower endpoint. For square M use its integer square root.',
 'moment':'For Y=a*X+b, E[Y**2]=a**2*Var(X)+(a*E[X]+b)**2. Retain both terms and substitute all numbers.',
 'poisson_scaled':'If X is Poisson with mean m and Y=a*X+b, Var(Y)=a**2*m. The constant offset b does not affect variance.'}


def main():
    from kaggle_secrets import UserSecretsClient
    records=read_json(ROOT/'records.json');data=build()
    qs={q['id']:q for s in ('train','validation') for q in data[s]}
    rejected=[r for rows in records.values() for r in rows if not r['accepted']]
    dest=ROOT/'focused';dest.mkdir(exist_ok=True)
    save_json(dest/'protocol.json',dict(ids=[r['id'] for r in rejected],max_attempts=1,cap=len(rejected),reminders=FOCUSED,reason='Initial multi-rule prompt produced genuine Poisson and interval errors; new focused supervision, not a retry claimed as original accuracy.',test_used=False))
    client=TeacherClient(dest,UserSecretsClient().get_secret('OPENROUTER_API_KEY'),len(rejected))
    # Sequential calls keep the persistent cap and ledger unambiguous.
    supplements=read_json(ROOT/'focused_supplements.json',{})
    for i,r in enumerate(rejected,1):
        q=qs[r['id']]
        if r['id'] not in supplements:
            rule=FOCUSED.get(q['category'],'Check the exact requested event and substitute every numerical parameter.')
            raw=client.call(q['id'],[dict(role='system',content=rule+' Return only Expression: and one fully substituted numeric expression. Keep operations unevaluated. No explanation or final answer.'),dict(role='user',content=prompt(q))],max_tokens=200)
            supplements[r['id']]=[dict(attempt='focused_0',raw=raw,judged=score(raw,q))]
            save_json(ROOT/'focused_supplements.json',supplements)
        if i%8==0:print('V15 FOCUSED',i,'/',len(rejected),flush=True)
    rows=teacher_rows()
    original=TeacherClient(ROOT,'',400).stats()
    summary=dict(original_usage=original,focused_usage=client.stats(),original_counts={s:dict(n=len(rs),accepted=sum(r['accepted'] for r in rs),first_attempt=sum(r['attempts'][0]['judged']['correct'] for r in rs)) for s,rs in records.items()},focused_accepted=sum(v[0]['judged']['correct'] for v in supplements.values()),training_counts={s:len(rs) for s,rs in rows.items()})
    save_json(ROOT/'summary.json',summary);package(ROOT)
    print('V15 PREPARATION COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':main()
