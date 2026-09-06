"""Short Llama solutions; independently checked numeric chains, durable costs."""
import json,threading
from pathlib import Path
from fractions import Fraction as F
from concurrent.futures import ThreadPoolExecutor
from flight_run_stats_v0_3 import TeacherClient,TEACHER,read_json,save_json,package
from stats_curriculum_v0_9 import build,calculate,prompt
from stats_v0_3_common import digest,parse_teacher,prompt_for,parse_answer
ROOT=Path('/kaggle/working/3beethoven_stats_v0_9')
CAP=500

class Client(TeacherClient):
    def __init__(self,key):
        super().__init__(ROOT,key,CAP);self.lock=threading.Lock();self.reserved=self.stats()['attempted_calls']
    def call(self,tag,*args,**kwargs):
        if not (self.root/'api_cache'/(tag+'.json')).exists():
            with self.lock:
                if self.reserved>=CAP:raise RuntimeError('Reserved call cap reached')
                self.reserved+=1
        return super().call(tag,*args,**kwargs)

def validate(obj,q):
    if not all(isinstance(obj.get(k),str) for k in ('rule','calculation','answer')):raise ValueError('String fields required')
    if not 15<=len(obj['rule'])<=240 or '\n' in obj['rule']:raise ValueError('Rule length/format')
    if len(obj['calculation'])>240 or '\n' in obj['calculation']:raise ValueError('Calculation length/format')
    if F(obj['answer'])!=F(q['answer']):raise ValueError('Answer mismatch')
    parts=obj['calculation'].split('=')
    if len(parts)<2 or len(parts)>5:raise ValueError('Need numerical equation with 2-5 equal parts')
    if parts[0].strip()==obj['answer'].strip() or not any(op in parts[0] for op in ('+','-','*','/','^','×','÷')):raise ValueError('Show a substituted calculation, not just the answer twice')
    if not all(calculate(s)==F(q['answer']) for s in parts):raise ValueError('False numeric equation')
    return obj

def target(obj):
    return 'Formula: '+obj['rule']+'\nCalculation: '+obj['calculation']+'\nAnswer: '+obj['answer']

def main():
    from kaggle_secrets import UserSecretsClient
    ROOT.mkdir(exist_ok=True);data=build()
    frozen=read_json(Path(__file__).resolve().parent.parent/'docs/STATS_V0_9_FROZEN_QUESTIONS.json');assert data==frozen
    protocol=dict(teacher=TEACHER,data_sha256=digest(data),max_calls=CAP,training_records=180,validation_records=24,test_records=48,acceptance='Every numeric equality exact; independent audit of all rule prose required before training',references='First attempt independent; failed attempts receive deterministic rule/expression/answer feedback',target='Three concise lines: Formula, Calculation, Answer; no misconceptions or option letters in numeric solution')
    old=read_json(ROOT/'generation_protocol.json');assert old is None or old==protocol
    save_json(ROOT/'generation_protocol.json',protocol)
    for split,qs in data.items():save_json(ROOT/(split+'_questions.json'),qs)
    client=Client(UserSecretsClient().get_secret('OPENROUTER_API_KEY'))
    def worker(q):
        path=ROOT/'accepted'/(q['id']+'.json');old=read_json(path)
        if old:assert old['question_sha256']==digest(q);return old
        error=''
        for attempt in range(5):
            instructions='Solve the statistics question. Return ONLY JSON with string fields rule, calculation, answer. rule: one correct concise rule under 200 characters. calculation: a numerical equality with substituted numbers, using only numbers, parentheses, + - * / ** and =; no variables, factorial or function names. Every part separated by = must equal the final result. answer: exact integer or fraction only. Keep all fields short. No option letters or misconception discussion.'
            user=q['question']
            if attempt:user+='\nReference rule: '+q['reference_rule']+'\nVerified expression: '+q['reference_expression']+' = '+q['answer']+'\nPrevious check: '+error
            if attempt>=3:
                instructions+=' Repair mode: reproduce the supplied verified numerical equation exactly as the calculation field. Do not expand it or add intermediate equalities. Use the supplied reference rule faithfully, and the exact reference answer. You must not replace any number.'
            tag=f'short_{q["id"]}_{attempt}'
            raw=client.call(tag,[dict(role='system',content=instructions),dict(role='user',content=user)],max_tokens=300,json_mode=True)
            try:obj=validate(parse_teacher(raw),q)
            except (ValueError,TypeError,SyntaxError,ZeroDivisionError,OverflowError) as exc:error=str(exc)[:160];continue
            record=dict(q,teacher_model=TEACHER,question_sha256=digest(q),reference_conditioned=attempt>0,cache_tag=tag,teacher_solution=obj,target=target(obj))
            save_json(path,record);return record
        raise RuntimeError('Numeric validation exhausted for '+q['id'])
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i,r in enumerate(pool.map(worker,data['train']+data['validation']),1):
                if i%12==0:print('V09 CORPUS',i,'/204',json.dumps(client.stats()),flush=True)
        for split in ('train','validation'):save_json(ROOT/(split+'_records.json'),[read_json(ROOT/'accepted'/(q['id']+'.json')) for q in data[split]])
        report=dict(protocol=protocol,usage=client.stats(),accepted=204,audit_status='Numeric validated; prose audit required')
        save_json(ROOT/'generation_complete.json',report)
        print('V09 CORPUS EXPORT',json.dumps(dict(report=report,train=read_json(ROOT/'train_records.json'),validation=read_json(ROOT/'validation_records.json'))),flush=True)
    finally:save_json(ROOT/'api_usage.json',client.stats());package(ROOT)

if __name__=='__main__':main()
