"""Focused rule-conditioned teacher responses; no numerical gold supplied."""
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from flight_run_stats_v0_3 import TeacherClient,save_json,read_json,package
from stats_curriculum_v0_16 import build,digest,prompt,RULES,score

ROOT=Path('/kaggle/working/3beethoven_stats_v0_16/teacher')

class Client(TeacherClient):
    def __init__(self,key):
        ROOT.mkdir(parents=True,exist_ok=True);super().__init__(ROOT,key,684)
        self.lock=threading.Lock();self.reserved=self.stats()['attempted_calls']
    def call(self,tag,*args,**kwargs):
        with self.lock:
            if not (ROOT/'api_cache'/(tag+'.json')).exists():
                s=self.stats()
                if self.reserved>=684 or s['reported_cost_usd']>=.30 or s['responses_without_cost']:
                    raise RuntimeError('Teacher bounded-budget gate')
                self.reserved+=1
        return super().call(tag,*args,**kwargs)

def main():
    from kaggle_secrets import UserSecretsClient
    data=build();repo=Path(__file__).resolve().parents[1]
    assert data==read_json(repo/'docs/STATS_V0_16_FROZEN_QUESTIONS.json')
    client=Client(UserSecretsClient().get_secret('OPENROUTER_API_KEY'))
    def worker(q):
        path=ROOT/'records'/(q['id']+'.json');prior=read_json(path)
        if prior:
            assert prior['question_sha256']==digest(q);return prior
        attempts=[]
        for a in range(3):
            system=RULES[q['task']]+' Return only a fully substituted numerical Expression. Keep operations unevaluated.'
            if a:system+=' Recheck the requested quantity and which numbers represent mean, variance, scale and offset.'
            raw=client.call(q['id']+'_'+str(a),[dict(role='system',content=system),dict(role='user',content=prompt(q))],max_tokens=200)
            judged=score(raw,q);attempts.append(dict(attempt=a,raw=raw,grade=judged))
            if judged['correct']:break
        row=dict(id=q['id'],question_sha256=digest(q),attempts=attempts,accepted=any(x['grade']['correct'] for x in attempts))
        save_json(path,row);return row
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,_ in enumerate(pool.map(worker,data['train']+data['validation']),1):
            if i%24==0:print('V16 TEACHER',i,'/228',flush=True)
    records={s:[read_json(ROOT/'records'/(q['id']+'.json')) for q in data[s]] for s in ('train','validation')}
    save_json(ROOT/'records.json',records)
    summary=dict(data_sha256=digest(data),usage=client.stats(),counts={s:dict(n=len(rs),accepted=sum(r['accepted'] for r in rs)) for s,rs in records.items()},reference_conditioned='Task-specific symbolic rules only; no numerical gold; test not sent')
    save_json(ROOT/'summary.json',summary);package(ROOT)
    print('V16 TEACHER COMPLETE',summary,flush=True)
    assert summary['counts']['train']['accepted']>=180 and summary['counts']['validation']['accepted']>=30

if __name__=='__main__':main()
