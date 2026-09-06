"""Audited shared Llama rule lines; preserve original per-example arithmetic."""
import json
from generate_stats_v0_9 import ROOT,Client,target,validated_solution
from flight_run_stats_v0_3 import read_json,save_json,package
from stats_v0_3_common import digest,parse_teacher
CHECKS={
 'poisson_count':'For a homogeneous Poisson process of rate lambda and duration t, count variance is lambda*t.',
 'poisson_scaled':'X is Poisson with mean lambda. Var(aX+b)=a^2*lambda. Constant b does not affect variance.',
 'expectation':'For mu=E[X] and v=Var(X), E[(aX+b)^2]=a^2*v+(a*mu+b)^2. No independence assumption is needed.',
 'uniform':'If T is continuous uniform on [l,u] and l<c<u, conditioning on T>c gives mean (c+u)/2. Name c the threshold and u the upper endpoint.',
 'type_i':'For n independent trials each with event probability p, probability exactly r occur is C(n,r)*p^r*(1-p)^(n-r). Define n,r,p.',
 'type_ii':'For two independent methods with miss probabilities b1,b2, exactly one detection has probability (1-b1)*b2+b1*(1-b2). This is not the inclusive union of detections.',
 'confidence':'Old interval has center C and half-width h. With confidence and population standard deviation fixed, multiplying sample size by k^2 gives new upper endpoint C+h/k, for k>0.'}

def key(r):return ('poisson_count' if r['parameters'][0]==0 else 'poisson_scaled') if r['category']=='poisson' else r['category']

def generate():
    from kaggle_secrets import UserSecretsClient
    client=Client(UserSecretsClient().get_secret('OPENROUTER_API_KEY'));rules={}
    for k,check in CHECKS.items():
        tag='audited_rule_v2_'+k
        raw=client.call(tag,[dict(role='system',content='Return only JSON with one string field rule. Write a complete English sentence of 60-220 characters, stating the mathematical relationship AND variable meanings. A bare formula or topic label is insufficient. Preserve all assumptions and distinguish exactly one from at least one. Use ASCII notation. No examples.'),dict(role='user',content=check)],max_tokens=200,json_mode=True)
        obj=parse_teacher(raw);assert isinstance(obj.get('rule'),str) and 40<=len(obj['rule'])<=320 and '\n' not in obj['rule']
        rules[k]=dict(rule=obj['rule'],cache_tag=tag,reference_check=check)
    save_json(ROOT/'rules.json',rules);save_json(ROOT/'api_usage.json',client.stats())
    print('V09 RULES EXPORT',json.dumps(dict(rules=rules,usage=client.stats())),flush=True)

def apply():
    rules=read_json(ROOT/'rules.json');approved=read_json(ROOT/'rules_approved.json')
    assert approved and approved['approved'] is True and approved['rules_sha256']==digest(rules)
    original=read_json(ROOT/'pre_rule_records.json')
    if original is None:
        original={s:read_json(ROOT/(s+'_records.json')) for s in ('train','validation')};save_json(ROOT/'pre_rule_records.json',original)
    out={}
    for split,rows in original.items():
        out[split]=[]
        for r in rows:
            source=validated_solution(read_json(ROOT/'api_cache'/(r['cache_tag']+'.json'))['text'],r)
            assert source==r['teacher_solution']
            rule=rules[key(r)];obj=dict(source,rule=rule['rule'])
            out[split].append(dict(r,original_teacher_solution=source,teacher_solution=obj,target=target(obj),rule_key=key(r),rule_cache_tag=rule['cache_tag']))
        save_json(ROOT/(split+'_records.json'),out[split])
    print('V09 FINAL CORPUS',json.dumps(dict(records=out,records_sha256=digest(out),rules=rules)),flush=True)
    package(ROOT)

def repair():
    from kaggle_secrets import UserSecretsClient
    client=Client(UserSecretsClient().get_secret('OPENROUTER_API_KEY'));rules=read_json(ROOT/'rules.json')
    fixes={
      'uniform':'State the conditional-uniform midpoint formula AND the condition l<c<u. Say the distribution is continuous.',
      'type_i':'You omitted the entire probability formula. Include P(R=r)=C(n,r)*p^r*(1-p)^(n-r), independence, and define n,r,p. Definitions alone are insufficient.',
      'type_ii':'Include independence and explicitly define b1,b2 as miss probabilities, followed by the exactly-one-detection formula. Do not say inclusive union.',
      'confidence':'State the fixed confidence level and population standard deviation assumptions. Say new upper endpoint C+h/k, old half-width h, center C, sample-size factor k^2 and k>0. Do not claim it decreases for all positive k.'}
    for k,feedback in fixes.items():
        tag='audited_rule_v3_'+k
        raw=client.call(tag,[dict(role='system',content='Return ONLY JSON with one string field rule, a complete mathematical statement under 240 characters. Include the formula and all requested assumptions and definitions; do not omit the formula to save space.'),dict(role='user',content=CHECKS[k]+'\nAudit correction: '+feedback)],max_tokens=200,json_mode=True)
        obj=parse_teacher(raw);assert isinstance(obj.get('rule'),str) and 40<=len(obj['rule'])<=320 and '\n' not in obj['rule']
        rules[k]=dict(rule=obj['rule'],cache_tag=tag,reference_check=CHECKS[k])
    save_json(ROOT/'rules.json',rules);save_json(ROOT/'api_usage.json',client.stats())
    print('V09 RULES EXPORT',json.dumps(dict(rules=rules,usage=client.stats())),flush=True)

if __name__=='__main__':
    import sys
    apply() if '--apply' in sys.argv else repair() if '--repair' in sys.argv else generate()
