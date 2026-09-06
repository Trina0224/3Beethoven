"""Llama numeric stages with exact checks and durable request provenance."""
import json,threading
from pathlib import Path
from fractions import Fraction as F
from concurrent.futures import ThreadPoolExecutor
from flight_run_stats_v0_3 import TeacherClient,read_json,save_json,package
from stats_v0_3_common import digest,parse_teacher
from stats_curriculum_v0_10 import build,calculate
ROOT=Path('/kaggle/working/3beethoven_stats_v0_10')
CAP=400

class Client(TeacherClient):
    def __init__(self,key):
        ROOT.mkdir(exist_ok=True);super().__init__(ROOT,key,CAP)
        self.lock=threading.Lock();self.reserved=self.stats()['attempted_calls']
    def call(self,tag,*args,**kwargs):
        if not (self.root/'api_cache'/(tag+'.json')).exists():
            with self.lock:
                if self.reserved>=CAP:raise RuntimeError('Call cap reached')
                self.reserved+=1
        return super().call(tag,*args,**kwargs)

def validate(obj,q):
    stages=obj.get('stages');answer=obj.get('answer')
    assert isinstance(stages,list) and 4<=len(stages)<=8
    assert all(isinstance(x,str) and len(x)<=180 and '\n' not in x for x in stages)
    assert isinstance(answer,str) and F(answer)==F(q['answer'])
    assert all(calculate(x)==F(q['answer']) for x in stages)
    assert any(x in stages[0] for x in ('*','+')) and len(set(stages))>=4
    assert len(' = '.join(stages))<=700
    return obj

def main():
    from kaggle_secrets import UserSecretsClient
    ROOT.mkdir(exist_ok=True);data=build()
    assert data==read_json(Path(__file__).resolve().parent.parent/'docs/STATS_V0_10_FROZEN_QUESTIONS.json')
    prior=read_json(ROOT/'data_sha.json');assert prior is None or prior==digest(data)
    save_json(ROOT/'data_sha.json',digest(data))
    client=Client(UserSecretsClient().get_secret('OPENROUTER_API_KEY'))
    def worker(q):
        dest=ROOT/'accepted'/(q['id']+'.json');previous=read_json(dest)
        if previous:assert previous['question_sha256']==digest(q);return previous
        error=''
        for attempt in range(3):
            system='Return JSON with stages (4-8 strings) and answer (one exact fraction string). Solve the probability problem by explicitly expanding powers, multiplying numerators and denominators, adding fractions and reducing. Every stage must be a numeric expression equal to the SAME final answer; no variable names, factorials, prose or equals signs within a stage. Use only numbers parentheses + - * / **. Start with fully substituted formula. Do not jump straight to final number.'
            user=q['question']
            if attempt:user+='\nVerified stages: '+json.dumps(q['reference_chain'])+'\nExact answer: '+q['answer']+'\nPrevious validation failed: '+error
            if attempt==2:system='Return ONLY the supplied JSON unchanged. This is reference-conditioned formatting repair, not independent calculation.';user=json.dumps(dict(stages=q['reference_chain'],answer=q['answer']))
            tag=q['id']+'_'+str(attempt)
            raw=client.call(tag,[dict(role='system',content=system),dict(role='user',content=user)],max_tokens=400,json_mode=True)
            try:obj=validate(parse_teacher(raw),q)
            except (AssertionError,ValueError,SyntaxError,TypeError,ZeroDivisionError,OverflowError) as exc:error=type(exc).__name__;continue
            row=dict(q,question_sha256=digest(q),cache_tag=tag,teacher_solution=obj,reference_conditioned=attempt>0,verbatim_reference_repair=attempt==2)
            save_json(dest,row);return row
        raise RuntimeError('Teacher validation exhausted '+q['id'])
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i,row in enumerate(pool.map(worker,data['train']+data['validation']),1):
                if i%16==0:print('V10 CORPUS',i,'/112',client.stats(),flush=True)
        records={s:[read_json(ROOT/'accepted'/(q['id']+'.json')) for q in data[s]] for s in ('train','validation')}
        save_json(ROOT/'teacher_records.json',records)
        print('V10 CORPUS EXPORT',json.dumps(dict(records=records,usage=client.stats(),records_sha256=digest(records))),flush=True)
    finally:save_json(ROOT/'api_usage.json',client.stats());package(ROOT)

if __name__=='__main__':main()
