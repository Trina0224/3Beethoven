"""Paired worked examples assembled from previously audited Llama answers.
Rejected abstract cards remain preserved but are never used by this builder.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from flight_run_stats_v0_3 import read_json,save_json,package
from generate_stats_v0_6 import ROOT,REPO,Client
from stats_v0_3_common import digest,prompt_for,parse_answer
from stats_holdout_v0_6 import build
PAIR={0:1,1:2,2:0}

def assemble(old):
    index={(r['family'],r['id'].rsplit('_',1)[1]):r for r in old['train']}
    rows=[]
    for r in old['train']:
        topic,part=r['family'].rsplit('_',1)
        other=index[(topic+'_'+str(PAIR[int(part)]),r['id'].rsplit('_',1)[1])]
        rows.append(dict(r,contrast_id=other['id'],contrast_question=other['question'],
            contrast_explanation=other['explanation'],source_sha256=digest(r),contrast_source_sha256=digest(other)))
    return rows

def main():
    from kaggle_secrets import UserSecretsClient
    ROOT.mkdir(exist_ok=True)
    existing=read_json(ROOT/'test_questions.json')
    if existing is not None and existing!=build():
        raise RuntimeError('Frozen test differs; refuse mixed runs')
    save_json(ROOT/'test_questions.json',build())
    old=read_json(REPO/'docs'/'STATS_V0_5_TEACHER_DATA.json')
    assert digest(old['train'])=='bcdae76a80519a037e3d7f3451800ff56a48c7f64fa8b26f13b993ef382a2c1f'
    save_json(ROOT/'source_records.json',old)
    rows=assemble(old)
    save_json(ROOT/'train_records.json',rows)
    save_json(ROOT/'validation_records.json',old['validation'])
    amendment=dict(method='Paired audited worked examples; abstract lesson cards rejected before training',
        primary_questions_unchanged=True,validation_unchanged=True,source_sha256=digest(old),
        train_sha256=digest(rows),pairs=[dict(id=r['id'],contrast_id=r['contrast_id']) for r in rows],
        target_provenance='Only original audited Llama explanations and misconceptions; pairing question is prompt context, headings are mechanical',
        rejected_abstract_cards='lessons/ and api_cache/ preserved, not used',
        audit='Original 204 answers independently read in v0.5; deterministic source equality rechecked for every pairing')
    save_json(ROOT/'pairing_protocol.json',amendment)
    client=Client(UserSecretsClient().get_secret('OPENROUTER_API_KEY'))
    teacher=read_json(ROOT/'teacher_test.json',[]);done={r['id'] for r in teacher}
    def call(q):
        raw=client.call('test_'+q['id'],[dict(role='user',content=prompt_for(q))],max_tokens=16)
        p=parse_answer(raw)
        return dict(id=q['id'],category=q['category'],expected=q['answer_letter'],predicted=p,correct=p==q['answer_letter'],raw=raw)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for r in pool.map(call,[q for q in build() if q['id'] not in done]):
                teacher.append(r);save_json(ROOT/'teacher_test.json',teacher)
        complete=dict(method=amendment['method'],api_usage=client.stats(),teacher_correct=sum(r['correct'] for r in teacher),
            teacher_n=48,pairing=amendment)
        save_json(ROOT/'generation_complete.json',complete)
        print('V06 PAIRED COMPLETE',__import__('json').dumps(complete),flush=True)
    finally:
        save_json(ROOT/'api_usage.json',client.stats());package(ROOT)

if __name__=='__main__':main()
