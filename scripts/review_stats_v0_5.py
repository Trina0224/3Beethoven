"""Targeted Llama revisions after an independent audit of all 204 targets.
Frozen questions, references, splits and 600-call ledger remain unchanged.
Original accepted targets and all API responses are retained.
"""
import json, shutil
from generate_stats_v0_5 import ROOT, ReservedClient, parse_output, valid_target
from flight_run_stats_v0_3 import read_json, save_json, package, TEACHER
from stats_v0_3_common import prompt_for, digest
from stats_curriculum_v0_5 import build

FEEDBACK = {
    "train_poisson_1_02": "For independent Poisson variables, summing means gives the same correct result as summing variances. Do not call that identity incorrect. Describe a genuinely wrong calculation.",
    "train_poisson_1_03": "For independent Poisson variables, summing means gives the same correct result as summing variances. Do not call that identity incorrect. Describe a genuinely wrong calculation.",
    "train_poisson_1_05": "For independent Poisson variables, summing means gives the same correct result as summing variances. Do not call that identity incorrect. Describe a genuinely wrong calculation.",
    "train_poisson_1_06": "For independent Poisson variables, summing means gives the same correct result as summing variances. Do not call that identity incorrect. Describe a genuinely wrong calculation.",
    "train_poisson_1_07": "For independent Poisson variables, summing means gives the same correct result as summing variances. Do not call that identity incorrect. Describe a genuinely wrong calculation.",
    "train_poisson_1_09": "For independent Poisson variables, summing means gives the same correct result as summing variances. Do not call that identity incorrect. Describe a genuinely wrong calculation.",
    "train_poisson_2_00": "Binomial variance is np(1-p), not the squared mean. Remove the false distribution claim.",
    "train_poisson_2_04": "Var(3X+6)=Var(3X)+Var(6) is valid because 6 is constant. Do not label this valid identity a mistake.",
    "train_poisson_2_07": "For a Poisson variable its mean equals its variance; using that mean in 9*Var(X) is valid. Explain a genuinely wrong scaling operation.",
    "train_poisson_2_08": "For a Poisson variable its mean equals its variance; using that mean in 9*Var(X) is valid. Explain a genuinely wrong scaling operation.",
    "train_poisson_2_09": "For a Poisson variable its mean equals its variance; using that mean in 9*Var(X) is valid. Explain a genuinely wrong scaling operation.",
    "train_expectation_0_01": "Linearity E[3X-c]=E[3X]-E[c]=3E[X]-c is valid, including for a constant. Do not call any of these identities incorrect. Clearly identify an actual omitted term or wrong sign.",
    "train_expectation_0_03": "Linearity E[3X-c]=E[3X]-E[c]=3E[X]-c is valid, including for a constant. Do not call any of these identities incorrect. Clearly identify an actual omitted term or wrong sign.",
    "train_expectation_0_04": "Linearity E[3X-c]=E[3X]-E[c]=3E[X]-c is valid, including for a constant. Do not call any of these identities incorrect. Clearly identify an actual omitted term or wrong sign.",
    "train_expectation_0_09": "Linearity E[3X-c]=E[3X]-E[c]=3E[X]-c is valid, including for a constant. Do not call any of these identities incorrect. Clearly identify an actual omitted term or wrong sign.",
    "train_expectation_2_03": "The intermediate equality 24+4=26 is false. Recompute both weighted terms and verify their sum.",
    "train_uniform_2_00": "For Uniform[0,b], P(X<x)=x/b is proportional to x for fixed b. Calling proportionality to the threshold or dividing by the maximum incorrect is false here. Give a genuinely incorrect probability calculation, clearly distinguished from the correct one.",
    "train_uniform_2_01": "For Uniform[0,b], P(X<x)=x/b is proportional to x for fixed b. Calling proportionality to the threshold or dividing by the maximum incorrect is false here. Give a genuinely incorrect probability calculation, clearly distinguished from the correct one.",
    "train_uniform_2_02": "For Uniform[0,b], P(X<x)=x/b is proportional to x for fixed b. Calling proportionality to the threshold or dividing by the maximum incorrect is false here. Give a genuinely incorrect probability calculation, clearly distinguished from the correct one.",
    "train_uniform_2_03": "For Uniform[0,b], P(X<x)=x/b is proportional to x for fixed b. Calling proportionality to the threshold or dividing by the maximum incorrect is false here. Give a genuinely incorrect probability calculation, clearly distinguished from the correct one.",
    "train_uniform_2_05": "For Uniform[0,b], P(X<x)=x/b is proportional to x for fixed b. Calling proportionality to the threshold or dividing by the maximum incorrect is false here. Give a genuinely incorrect probability calculation, clearly distinguished from the correct one.",
    "train_uniform_2_06": "For Uniform[0,b], P(X<x)=x/b is proportional to x for fixed b. Calling proportionality to the threshold or dividing by the maximum incorrect is false here. Give a genuinely incorrect probability calculation, clearly distinguished from the correct one.",
    "train_uniform_2_07": "For Uniform[0,b], P(X<x)=x/b is proportional to x for fixed b. Calling proportionality to the threshold or dividing by the maximum incorrect is false here. Give a genuinely incorrect probability calculation, clearly distinguished from the correct one.",
    "train_uniform_2_08": "For Uniform[0,b], P(X<x)=x/b is proportional to x for fixed b. Calling proportionality to the threshold or dividing by the maximum incorrect is false here. Give a genuinely incorrect probability calculation, clearly distinguished from the correct one.",
    "train_type_ii_2_05": "The studies are independent. Adding rejection probabilities fails because the events overlap, not because of dependence. Correct the misconception.",
    "train_confidence_1_01": "Reducing width BY one third means keeping two thirds; this question asks reducing width TO one third. Use precise wording consistent with the question.",
    "validation_confidence_3_00": "Here standard error and critical value both happen to equal 2. Explain why blindly equating them is not a valid general method without falsely denying their numerical equality in this case."
}

def main():
    from kaggle_secrets import UserSecretsClient
    client = ReservedClient(ROOT, UserSecretsClient().get_secret('OPENROUTER_API_KEY'))
    backup = ROOT/'before_content_review'
    backup.mkdir(exist_ok=True)
    for name in ('train_records.json','validation_records.json','generation_complete.json'):
        if not (backup/name).exists(): shutil.copy2(ROOT/name,backup/name)
    data=build()
    items={r['id']:r for split in ('train','validation') for r in data[split]}
    report=[]
    try:
        for ident, feedback in FEEDBACK.items():
            item=items[ident]
            path=ROOT/'accepted'/(ident+'.json')
            old=read_json(path)
            if not (backup/(ident+'.json')).exists(): save_json(backup/(ident+'.json'),old)
            if old.get('content_review_revision'):
                report.append(dict(id=ident,feedback=feedback,target_cache_tag=old['target_cache_tag']))
                continue
            for attempt in range(2):
                tag=f'content_revision_{ident}_{attempt}'
                user=(prompt_for(item,'explain').split('\n\nChoose A, B, C, or D')[0]
                    +'\nReference: '+item['reference_reason']+' Correct choice: '+item['answer_letter']
                    +'\nIndependent content audit feedback: '+feedback
                    +'\nRevise the explanation and common mistake. Check every arithmetic equality.')
                cached_recovery = ident == 'train_expectation_2_03'
                if cached_recovery:
                    # This earlier paid response contains correct prose and an
                    # explicit corrected label; preserve and review it verbatim.
                    tag='generate_train_expectation_2_03_2'
                raw=read_json(ROOT/'api_cache'/(tag+'.json'))['text'] if cached_recovery else client.call(tag,[
                    dict(role='system',content='Return only JSON string fields answer_letter, explanation, common_mistake. Explain the correct calculation in 2 concise sentences (at least 40 characters). Name a genuinely wrong misconception (at least 15 characters). Never label a valid identity as incorrect.'),
                    dict(role='user',content=user)],json_mode=True)
                try: target=parse_output(raw)
                except (ValueError,TypeError): continue
                if not valid_target(target,item): continue
                review_tag=f'review_content_{ident}_{attempt}' if not cached_recovery else 'review_content_recovered_expectation_2_03'
                review_raw=client.call(review_tag,[
                    dict(role='system',content='Check every calculation and misconception. Return only JSON valid (boolean), answer_letter, reason (one sentence under 160 characters). A correct letter is insufficient. Reject false identities and valid methods mislabeled as wrong.'),
                    dict(role='user',content=user+'\nCandidate: '+json.dumps(target))],max_tokens=250,json_mode=True)
                review=parse_output(review_raw)
                save_json(ROOT/'reviews'/(review_tag.removeprefix('review_')+'.json'),review)
                if review.get('valid') is not True or review.get('answer_letter')!=item['answer_letter']: continue
                revised=dict(old,explanation=target['explanation'],common_mistake=target['common_mistake'],
                    reference_conditioned=True,content_review_revision=True,target_cache_tag=tag,review_tag=review_tag)
                assert revised['question_sha256']==digest(item) and revised['teacher_model']==TEACHER
                save_json(path,revised)
                report.append(dict(id=ident,feedback=feedback,target_cache_tag=tag))
                print('V05 CONTENT REVISED',ident,json.dumps(target),flush=True)
                break
            else: raise RuntimeError('Content revision failed: '+ident)
        for split in ('train','validation'):
            save_json(ROOT/(split+'_records.json'),[read_json(ROOT/'accepted'/(r['id']+'.json')) for r in data[split]])
        audit=dict(audited_records=204,method='Assistant read all explanations and misconceptions against mathematical references; flagged targets revised by Llama only; same-teacher approval is not independent proof',
            revisions=report,originals='before_content_review/',api_usage=client.stats())
        save_json(ROOT/'content_review.json',audit)
        complete=read_json(ROOT/'generation_complete.json')
        complete.update(api_usage=client.stats(),content_review=audit,
            reference_conditioned=sum(read_json(ROOT/'accepted'/(r['id']+'.json'))['reference_conditioned'] for r in items.values()))
        save_json(ROOT/'generation_complete.json',complete)
        print('V05 CONTENT COMPLETE',json.dumps(client.stats()),flush=True)
    finally:
        shutil.copy2(__file__,ROOT/'source'/'review_stats_v0_5.py')
        save_json(ROOT/'api_usage.json',client.stats())
        package(ROOT)

if __name__=='__main__': main()
