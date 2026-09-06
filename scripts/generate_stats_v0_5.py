"""Four bounded request workers, durable request ledger, teacher-only targets."""
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from flight_run_stats_v0_3 import TeacherClient,TEACHER,read_json,save_json,package
from stats_v0_3_common import prompt_for,parse_teacher,digest,parse_answer
from stats_curriculum_v0_5 import build,audit
from stats_holdout_v1 import questions as old1
from stats_holdout_v2 import questions as old2
from stats_v0_3_common import make_curriculum

ROOT=Path('/kaggle/working/3beethoven_stats_v0_5')
MAX_CALLS=600


class ReservedClient(TeacherClient):
    def __init__(self,root,key):
        super().__init__(root,key,MAX_CALLS)
        self.lock=threading.Lock(); self.reserved=self.stats()['attempted_calls']
    def call(self,tag,*args,**kwargs):
        # A reservation also covers calls not yet visible in the durable ledger.
        # Each tag is owned by a single worker. Cache hits need no reservation.
        if not (self.root/'api_cache'/(tag+'.json')).exists():
            with self.lock:
                if self.reserved>=MAX_CALLS: raise RuntimeError('Global reserved call limit reached')
                self.reserved+=1
        return super().call(tag,*args,**kwargs)


def valid_target(obj,item):
    return (obj.get('answer_letter')==item['answer_letter'] and
        all(isinstance(obj.get(k),str) and low<=len(obj[k])<=high
            for k,low,high in (('explanation',40,1200),('common_mistake',15,500))))


def parse_output(raw):
    cleaned=raw.strip()
    prefix=re.match(r'^Answer:\s*([ABCD])\s*\n',cleaned)
    if prefix:
        body=cleaned[prefix.end():].strip()
        if body.startswith('{'):
            obj=json.loads(body)
            if obj.get('answer_letter')!=prefix.group(1): raise ValueError('Conflicting answer prefix')
            return obj
        parts=re.split(r'\n(?:Common mistake|Common misconception):\s*',body,maxsplit=1,flags=re.I)
        if len(parts)==2:
            return dict(answer_letter=prefix.group(1),explanation=parts[0].strip(),common_mistake=parts[1].strip())
    return parse_teacher(raw)


def main():
    from kaggle_secrets import UserSecretsClient
    ROOT.mkdir(exist_ok=True)
    data=build(); checked=audit(make_curriculum()+old1()+old2())
    protocol=dict(version='stats-v0.5',splits=checked,teacher=TEACHER,max_calls=MAX_CALLS,
        workers=4,teacher_test_mode='letter16 original order only',
        acceptance='Exact reference label plus teacher content review; same-teacher review is not independent proof',
        references='Deterministic mathematical references; explanations and misconceptions are Llama outputs only')
    prior=read_json(ROOT/'generation_protocol.json')
    if prior and prior!=protocol: raise RuntimeError('Frozen protocol mismatch')
    save_json(ROOT/'generation_protocol.json',protocol)
    for split,rows in data.items(): save_json(ROOT/(split+'_questions.json'),rows)
    (ROOT/'accepted').mkdir(exist_ok=True)
    (ROOT/'reviews').mkdir(exist_ok=True)
    (ROOT/'source').mkdir(exist_ok=True)
    import shutil
    for name in ('generate_stats_v0_5.py','stats_curriculum_v0_5.py','flight_run_stats_v0_3.py','stats_v0_3_common.py'):
        shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    client=ReservedClient(ROOT,UserSecretsClient().get_secret('OPENROUTER_API_KEY'))
    stopping=threading.Event()
    def worker(item):
        if stopping.is_set(): return None
        out=ROOT/'accepted'/(item['id']+'.json')
        existing=read_json(out)
        if existing:
            if existing['question_sha256']!=digest(item): raise RuntimeError('Cached item mismatch')
            return existing
        feedback=''
        try:
            for attempt in range(3):
                if stopping.is_set(): return None
                system='Solve independently. Return only JSON string fields answer_letter, explanation, common_mistake. Explain the decisive calculation in 2-3 concise sentences. Describe a genuinely incorrect misconception explicitly as incorrect. Keep explanation under 900 characters and misconception under 300.'
                user=prompt_for(item,'explain').split('\n\nChoose A, B, C, or D')[0]+'\nReturn only the requested JSON object.'
                if attempt:
                    user+='\nReference check: '+item['reference_reason']+' Correct choice: '+item['answer_letter']+'.\nReview feedback: '+feedback[:500]
                tag=f"generate_{item['id']}_{attempt}"
                cached=read_json(ROOT/'api_cache'/(tag+'.json'))
                # Existing paid responses retain their original request/provenance.
                raw=cached['text'] if cached else client.call(tag,[dict(role='system',content=system),dict(role='user',content=user)],json_mode=True)
                try: obj=parse_output(raw)
                except (ValueError,TypeError):
                    feedback='Previous response was not the requested JSON schema.'; continue
                if not valid_target(obj,item):
                    feedback='The label disagreed with the reference or explanation schema failed.'; continue
                if stopping.is_set(): return None
                review_prompt=(prompt_for(item,'explain').split('\n\nChoose A, B, C, or D')[0]+
                    '\nReference: '+item['reference_reason']+' Reference choice: '+item['answer_letter']+'.'+
                    '\nCandidate answer: '+json.dumps(obj))
                review_system='You are reviewing a candidate, not answering a chat prompt. Check every calculation, conclusion and misconception. Return only JSON with valid (boolean), answer_letter (correct option), and reason (one sentence under 160 characters). Mark valid false if any mathematical claim is wrong or a valid identity is called a mistake. A matching answer letter alone is insufficient.'
                review_tag=f"review_v2_{item['id']}_{attempt}"
                review_raw=client.call(review_tag,[dict(role='system',content=review_system),dict(role='user',content=review_prompt)],max_tokens=250,json_mode=True)
                try: review=parse_output(review_raw)
                except (ValueError,TypeError): review=dict(valid=False,reason='Invalid review JSON')
                save_json(ROOT/'reviews'/(review_tag.removeprefix('review_')+'.json'),review)
                if review.get('valid') is True and review.get('answer_letter')==item['answer_letter']:
                    accepted=dict(item,explanation=obj['explanation'],common_mistake=obj['common_mistake'],
                        teacher_model=TEACHER,question_sha256=digest(item),reference_conditioned=attempt>0,
                        accepted_attempt=attempt,review_tag=review_tag)
                    save_json(out,accepted)
                    return accepted
                feedback=str(review.get('reason','Content review rejected response'))
            raise RuntimeError('Teacher validation exhausted for '+item['id'])
        except Exception:
            stopping.set(); raise
    print('V05 GENERATION FROZEN',json.dumps(protocol),flush=True)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(worker,r) for r in data['train']+data['validation']]
            completed=0
            for future in as_completed(futures):
                if future.result() is not None: completed+=1
                if completed and completed%12==0:
                    print('V05 CORPUS',completed,'/204',json.dumps(client.stats()),flush=True)
        for split in ('train','validation'):
            rows=[read_json(ROOT/'accepted'/(r['id']+'.json')) for r in data[split]]
            if any(r is None for r in rows): raise RuntimeError('Incomplete accepted corpus')
            save_json(ROOT/(split+'_records.json'),rows)
        # Test answers are recorded separately and never read by training builder.
        teacher=read_json(ROOT/'teacher_test.json',[]); done={r['id'] for r in teacher}
        for item in data['test']:
            if item['id'] in done: continue
            raw=client.call('test_'+item['id'],[dict(role='user',content=prompt_for(item))],max_tokens=16)
            pred=parse_answer(raw)
            teacher.append(dict(id=item['id'],category=item['category'],expected=item['answer_letter'],
                                predicted=pred,correct=pred==item['answer_letter'],raw=raw))
            save_json(ROOT/'teacher_test.json',teacher)
            if len(teacher)%12==0: print('V05 TEACHER TEST',len(teacher),'/36',flush=True)
        report=dict(protocol=protocol,api_usage=client.stats(),accepted=204,
            reference_conditioned=sum(read_json(ROOT/'accepted'/(r['id']+'.json'))['reference_conditioned'] for r in data['train']+data['validation']),
            teacher_correct=sum(r['correct'] for r in teacher),teacher_n=36)
        save_json(ROOT/'generation_complete.json',report)
        print('V05 GENERATION COMPLETE',json.dumps(report),flush=True)
    finally:
        save_json(ROOT/'api_usage.json',client.stats()); package(ROOT)


if __name__=='__main__': main()
