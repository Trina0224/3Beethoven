"""Llama lesson cards from training families only; retained paid provenance."""
import json,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from flight_run_stats_v0_3 import TeacherClient,TEACHER,read_json,save_json,package
from stats_v0_3_common import digest,prompt_for,parse_answer,parse_teacher
from stats_holdout_v0_6 import build,audit

ROOT=Path('/kaggle/working/3beethoven_stats_v0_6')
REPO=Path(__file__).resolve().parent.parent
CAP=180
CHECKS={
 'expectation_0':'Linearity of expectation always holds without independence. Contrast scaling a variable with squaring a variable; never claim dependence breaks linearity.',
 'expectation_1':'Second moment equals variance plus mean squared. Mean squared equals second moment only when variance is zero.',
 'expectation_2':'Equal-weight averaging is a special case of weighted averaging, not an alternative that invalidates weighted sums.',
 'confidence_0':'Finite two-sided interval width is twice margin of error. A one-sided confidence interval has an infinite endpoint; do not give it a finite total width. Contrast total width versus margin of error instead.',
 'confidence_1':'If new width is r times old width, new sample size is 1/(r*r) times old size. Define r clearly. Contrast sample size scaling with standard deviation scaling.',
 'confidence_2':'Width scales linearly with population standard deviation and inversely with square root of sample size.',
 'poisson_0':'Increasing observation duration scales count variance linearly; multiplying the realized random count scales variance quadratically.',
 'poisson_1':'Independent variances add; for dependent variables covariance must be included. Poisson mean equals variance.',
 'poisson_2':'Var(aX+b)=a*a*Var(X). A constant shift never adds to variance; Poisson mean equals variance.',
 'uniform_0':'A continuous uniform mean is the midpoint; variance uses squared interval length divided by 12.',
 'uniform_1':'A continuous uniform variance uses squared interval length divided by 12; shifting both endpoints equally does not change variance.',
 'uniform_2':'Interval probability is favorable length divided by total length. Probability at any single point is zero.',
 'type_i_0':'Expected false rejections is sum of per-test probabilities even with dependence; probability of at least one is a different quantity.',
 'type_i_1':'Union bound requires no independence and is an upper bound, not generally an exact probability.',
 'type_i_2':'Independence permits multiplying probabilities of avoiding rejection. At least one is the complement of none.',
 'type_ii_0':'At a fixed alternative, power=1-beta. Alpha is defined under a true null and is not one minus beta.',
 'type_ii_1':'Expected misses=sum of beta over studies, no independence needed. Expected detections=sum of power.',
 'type_ii_2':'For independent studies both miss with probability beta squared; at least one detects is its complement. Both detecting is power squared.'}

class Client(TeacherClient):
    def __init__(self,key):
        super().__init__(ROOT,key,CAP)
        self.lock=threading.Lock();self.reserved=self.stats()['attempted_calls']
    def call(self,tag,*args,**kwargs):
        if not (ROOT/'api_cache'/(tag+'.json')).exists():
            with self.lock:
                if self.reserved>=CAP: raise RuntimeError('Reserved request cap reached')
                self.reserved+=1
        return super().call(tag,*args,**kwargs)

def main():
    from kaggle_secrets import UserSecretsClient
    ROOT.mkdir(exist_ok=True)
    old=read_json(REPO/'docs'/'STATS_V0_5_TEACHER_DATA.json')
    assert digest(old['train'])=='bcdae76a80519a037e3d7f3451800ff56a48c7f64fa8b26f13b993ef382a2c1f'
    protocol=dict(source_train_sha256=digest(old['train']),source_validation_sha256=digest(old['validation']),
        test=audit(old['train']+old['validation']),teacher=TEACHER,max_calls=CAP,
        design='18 Llama lesson cards appended to the same 180 worked examples; replace misconception with explicit contrast; validation records unchanged')
    prior=read_json(ROOT/'generation_protocol.json')
    if prior and prior!=protocol: raise RuntimeError('Protocol mismatch')
    save_json(ROOT/'generation_protocol.json',protocol)
    save_json(ROOT/'test_questions.json',build())
    save_json(ROOT/'source_records.json',old)
    client=Client(UserSecretsClient().get_secret('OPENROUTER_API_KEY'))
    bad=ROOT/'lessons'/'expectation_0.json'
    if bad.exists() and read_json(bad)['target_cache_tag']=='lesson_expectation_0_2':
        save_json(ROOT/'rejected_cards'/'expectation_0_initial.json',read_json(bad));bad.unlink()
    families=sorted({r['family'] for r in old['train']})
    def worker(family):
        path=ROOT/'lessons'/(family+'.json')
        if path.exists(): return read_json(path)
        examples=[r for r in old['train'] if r['family']==family]
        context='\n'.join(r['question']+' Reference: '+r['reference_reason'] for r in (examples[0],examples[-1]))
        feedback=CHECKS[family]
        for attempt in range(3):
            tag=f'lesson_v2_{family}_{attempt}'
            raw=client.call(tag,[
                dict(role='system',content='Teach the shared mathematical method, not memorized option letters. Return JSON lesson and contrast, both strings. lesson: a short rule and 2 numbered reusable calculation steps. contrast: one closely related situation requiring a DIFFERENT operation, explaining the difference precisely. Under 100 words total. Use symbolic formulas. Do not call valid identities mistakes.'),
                dict(role='user',content=context+'\n'+feedback)],json_mode=True)
            try: obj=parse_teacher(raw)
            except (ValueError,TypeError): feedback='Use the requested JSON schema.';continue
            if not all(isinstance(obj.get(k),str) and 35<=len(obj[k])<=1000 for k in ('lesson','contrast')):
                feedback='Both strings need a meaningful concise teaching explanation.';continue
            rt=f'review_v2_{family}_{attempt}'
            reviewraw=client.call(rt,[
                dict(role='system',content='You are a JSON validity reviewer. Do not solve example questions. Return ONLY one JSON object with valid boolean and reason under 120 characters. Reject any false mathematical claim.'),
                dict(role='user',content='Reference checks: '+CHECKS[family]+'\nReview this lesson and contrast: '+json.dumps(obj))],max_tokens=180,json_mode=True)
            try: review=parse_teacher(reviewraw)
            except (ValueError,TypeError): review=dict(valid=False,reason='Review was not valid JSON')
            if review.get('valid') is True:
                out=dict(family=family,lesson=obj['lesson'],contrast=obj['contrast'],teacher_model=TEACHER,
                    target_cache_tag=tag,review_cache_tag=rt,review=review)
                save_json(path,out);print('V06 LESSON',json.dumps(out),flush=True);return out
            feedback=CHECKS[family]+' '+str(review.get('reason','Recheck mathematics'))
        raise RuntimeError('Lesson failed: '+family)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool: cards=list(pool.map(worker,families))
        cardmap={r['family']:r for r in cards}
        records=[]
        for r in old['train']:
            c=cardmap[r['family']]
            records.append(dict(r,explanation=c['lesson']+'\nWorked calculation: '+r['explanation'],
                common_mistake=c['contrast'],lesson_source=c['target_cache_tag'],original_target_sha256=digest(r)))
        save_json(ROOT/'train_records.json',records)
        save_json(ROOT/'validation_records.json',old['validation'])
        save_json(ROOT/'lessons.json',cards)
        teacher=read_json(ROOT/'teacher_test.json',[]); done={r['id'] for r in teacher}
        def test(q):
            raw=client.call('test_'+q['id'],[dict(role='user',content=prompt_for(q))],max_tokens=16)
            pred=parse_answer(raw)
            return dict(id=q['id'],category=q['category'],expected=q['answer_letter'],predicted=pred,correct=pred==q['answer_letter'],raw=raw)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for r in pool.map(test,[q for q in build() if q['id'] not in done]):
                teacher.append(r);save_json(ROOT/'teacher_test.json',teacher)
        report=dict(protocol=protocol,api_usage=client.stats(),teacher_correct=sum(r['correct'] for r in teacher),teacher_n=48)
        save_json(ROOT/'generation_complete.json',report)
        print('V06 GENERATION COMPLETE',json.dumps(report),flush=True)
        print('V06 LESSON EXPORT',json.dumps(cards),flush=True)
    finally:
        save_json(ROOT/'api_usage.json',client.stats());package(ROOT)

if __name__=='__main__':main()
