"""Rule-scaffolded teacher curriculum, no test targets or gold answers supplied."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from flight_run_stats_v0_3 import TeacherClient,read_json,save_json,package
from stats_curriculum_v0_15 import build,digest,prompt,score,RULES

ROOT=Path('/kaggle/working/3beethoven_stats_v0_15/teacher')
DATA_SHA='d7df943fa318ffbaeffabd12834a8f345c54f71c48f8c8cc1cbef8c78ba55f1f'


class Client(TeacherClient):
    def __init__(self,key):
        ROOT.mkdir(parents=True,exist_ok=True);super().__init__(ROOT,key,400)
        self.lock=threading.Lock();self.reserved=self.stats()['attempted_calls']
    def call(self,tag,*args,**kwargs):
        with self.lock:
            if not (self.root/'api_cache'/(tag+'.json')).exists():
                usage=self.stats()
                if self.reserved>=400 or usage['reported_cost_usd']>=.25 or usage['responses_without_cost']:
                    raise RuntimeError('Teacher budget/cost reporting gate')
                self.reserved+=1
        return super().call(tag,*args,**kwargs)


def teacher_rows(repo=None):
    data=build();rows={};source=read_json(ROOT/'records.json')
    if not source:raise RuntimeError('Prepare teacher responses first')
    for split in ('train','validation'):
        qs={q['id']:q for q in data[split]};rows[split]=[]
        supplements=read_json(ROOT/'focused_supplements.json',{})
        for r in source[split]:
            q=qs[r['id']]
            assert r['question_sha256']==digest(q)
            for a in r['attempts']+supplements.get(r['id'],[]):
                judged=score(a['raw'],q)
                if judged['correct']:
                    rows[split].append(dict(source_id=q['id'],prompt=prompt(q),target='Expression: '+judged['normalized_expression'],teacher_raw=a['raw'],selected_attempt=a['attempt'],source='rule-scaffolded 70B response'))
                    break
    assert len(rows['train'])>=128 and len(rows['validation'])>=24
    # Replay only historical TRAINING examples, never evaluation answers.
    repo=repo or Path(__file__).resolve().parents[1]
    old=read_json(repo/'docs/STATS_V0_14_VERIFIED_DISTILLATION.json')['train']
    for kind in ('poisson_time','poisson_scaled','moment','uniform_time','binomial','exactly_one','at_least_one','interval'):
        selected=[r for r in old if f'_{kind}_' in r['source_id']][:4]
        assert len(selected)==4
        rows['train'].extend(dict(r,source='replayed verified v14 teacher training example') for r in selected)
    return rows


def main():
    from kaggle_secrets import UserSecretsClient
    data=build();assert digest(data)==DATA_SHA
    repo=Path(__file__).resolve().parents[1]
    assert data==read_json(repo/'docs/STATS_V0_15_FROZEN_QUESTIONS.json')
    client=Client(UserSecretsClient().get_secret('OPENROUTER_API_KEY'))
    save_json(ROOT/'protocol.json',dict(data_sha256=DATA_SHA,rules=RULES,max_calls=400,max_attempts=2,reference_conditioned='general symbolic formulas only',test_sent=False))
    def worker(q):
        path=ROOT/'review_records'/(q['id']+'.json');prior=read_json(path)
        if prior:
            assert prior['question_sha256']==digest(q);return prior
        attempts=[]
        for attempt in range(2):
            system=RULES+(' Recheck the requested quantity, units and complete numerical substitution.' if attempt else '')
            raw=client.call(q['id']+'_'+str(attempt),[dict(role='system',content=system),dict(role='user',content=prompt(q))],max_tokens=220)
            judged=score(raw,q);attempts.append(dict(attempt=attempt,raw=raw,judged=judged))
            if judged['correct']:break
        row=dict(id=q['id'],category=q['category'],question_sha256=digest(q),attempts=attempts,accepted=any(a['judged']['correct'] for a in attempts))
        save_json(path,row);return row
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i,_ in enumerate(pool.map(worker,data['train']+data['validation']),1):
                if i%16==0:print('V15 TEACHER',i,'/200',flush=True)
        records={s:[read_json(ROOT/'review_records'/(q['id']+'.json')) for q in data[s]] for s in ('train','validation')}
        save_json(ROOT/'records.json',records)
        rows=teacher_rows(repo)
        summary=dict(usage=client.stats(),counts={s:dict(candidates=len(rs),accepted=sum(r['accepted'] for r in rs),first_attempt=sum(r['attempts'][0]['judged']['correct'] for r in rs)) for s,rs in records.items()},training_counts={s:len(rs) for s,rs in rows.items()})
        save_json(ROOT/'summary.json',summary)
        print('V15 TEACHER COMPLETE',json.dumps(summary),flush=True)
    finally:package(ROOT)


if __name__=='__main__':main()
