"""Independent Llama teacher responses; accepted targets are NEVER gold copies."""
import json
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from flight_run_stats_v0_3 import TeacherClient,read_json,save_json,package
from stats_curriculum_v0_14 import build,prompt,score,digest

ROOT=Path('/kaggle/working/3beethoven_stats_v0_14_teacher')
DATA_SHA='78da1ed6f18c5068e6dd0cc2608be16292c9eaa67c98d3864ca1db7691430d38'
CAP=320


class Client(TeacherClient):
    def __init__(self,key):
        ROOT.mkdir(exist_ok=True);super().__init__(ROOT,key,CAP)
        self.lock=threading.Lock();self.reserved=self.stats()['attempted_calls']

    def call(self,tag,*args,**kwargs):
        with self.lock:
            if not (self.root/'api_cache'/(tag+'.json')).exists():
                usage=self.stats()
                if self.reserved>=CAP or usage['reported_cost_usd']>=1 or usage['responses_without_cost']:
                    raise RuntimeError('Teacher budget or cost-reporting gate reached')
                self.reserved+=1
        return super().call(tag,*args,**kwargs)


def main():
    from kaggle_secrets import UserSecretsClient
    data=build();assert digest(data)==DATA_SHA
    assert data==read_json(Path(__file__).resolve().parents[1]/'docs/STATS_V0_14_FROZEN_QUESTIONS.json')
    ROOT.mkdir(exist_ok=True)
    save_json(ROOT/'protocol.json',dict(data_sha256=DATA_SHA,cap=CAP,
        method='Independent response distillation preparation',reference_conditioned=False,
        evaluation_questions_sent_to_teacher=False,max_attempts=2,
        acceptance='Exact bindings, executable expression and conservative structure; alternatives preserved for review'))
    client=Client(UserSecretsClient().get_secret('OPENROUTER_API_KEY'))

    def worker(q):
        dest=ROOT/'review_records'/(q['id']+'.json')
        prior=read_json(dest)
        if prior:
            assert prior['question_sha256']==digest(q)
            return prior
        attempts=[]
        for attempt in range(2):
            system=('Solve independently. Return only the requested two lines. Use the exact binding names. '
                    'Substitute the question numbers into a complete expression without evaluating arithmetic, '
                    'powers, complements, combinations, or conversion factors. Do not output a final answer.')
            if attempt:system+=' Recheck the event, units and formatting carefully.'
            raw=client.call(q['id']+'_'+str(attempt),[dict(role='system',content=system),dict(role='user',content=prompt(q))],max_tokens=200)
            judged=score(raw.strip(),q)
            attempts.append(dict(attempt=attempt,raw=raw,judged=judged))
            if judged['correct']:break
        accepted=attempts[-1]['judged']['correct']
        row=dict(id=q['id'],category=q['category'],question_sha256=digest(q),question=q['question'],
                 prompt=prompt(q),accepted=accepted,attempts=attempts,reference_conditioned=False,
                 target=attempts[-1]['raw'].strip() if accepted else None)
        save_json(dest,row);return row

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i,row in enumerate(pool.map(worker,data['train']+data['validation']),1):
                if i%8==0:print('V14 TEACHER',i,'/160',json.dumps(client.stats()),flush=True)
        records={s:[read_json(ROOT/'review_records'/(q['id']+'.json')) for q in data[s]] for s in ('train','validation')}
        save_json(ROOT/'teacher_records.json',records)
        summary=dict(usage=client.stats(),counts={s:dict(total=len(rs),accepted=sum(r['accepted'] for r in rs),first_attempt_accepted=sum(r['attempts'][0]['judged']['correct'] for r in rs)) for s,rs in records.items()},
                     review='Pending content review. No student training started by this script.',data_sha256=DATA_SHA)
        save_json(ROOT/'summary.json',summary)
        print('V14 TEACHER COMPLETE',json.dumps(summary),flush=True)
    finally:
        save_json(ROOT/'api_usage.json',client.stats());package(ROOT)


if __name__=='__main__':main()
