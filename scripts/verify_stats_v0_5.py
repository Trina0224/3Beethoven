"""Independent response and artifact checks; export only public experiment data."""
import hashlib
import json
import zipfile
from pathlib import Path
from flight_run_stats_v0_3 import read_json,save_json
from stats_curriculum_v0_5 import build
from stats_v0_3_common import digest,parse_answer
from generate_stats_v0_5 import parse_output

ROOT=Path('/kaggle/working/3beethoven_stats_v0_5')


def corpus():
    frozen=build(); export=dict(generation=read_json(ROOT/'generation_complete.json'))
    for split in ('train','validation'):
        rows=read_json(ROOT/(split+'_records.json'))
        assert len(rows)==len(frozen[split])
        for r,q in zip(rows,frozen[split]):
            assert r['question_sha256']==digest(q)
            tag=r.get('target_cache_tag',f"generate_{r['id']}_{r['accepted_attempt']}")
            cache=read_json(ROOT/'api_cache'/(tag+'.json'))
            target=parse_output(cache['text'])
            assert all(r[k]==target[k] for k in ('answer_letter','explanation','common_mistake'))
            assert r['answer_letter']==q['answer_letter']
            review=parse_output(read_json(ROOT/'api_cache'/(r['review_tag']+'.json'))['text'])
            assert review.get('valid') is True and review['answer_letter']==q['answer_letter']
        export[split]=rows
    export['teacher_test']=read_json(ROOT/'teacher_test.json')
    assert len(export['teacher_test'])==36
    print('V05 CORPUS VERIFIED',len(export['train']),len(export['validation']),flush=True)
    print('V05 CORPUS EXPORT',json.dumps(export,ensure_ascii=False),flush=True)
    return export


def verify():
    export={'summary':read_json(ROOT/'summary.json'),
            'environment':read_json(ROOT/'environment.json'),
            'trainer_log':read_json(ROOT/'trainer_log.json')}
    for label,n in (('new',36),('old',60)):
        qs={r['id']:r for r in read_json(ROOT/(label+'_benchmark.json'))}
        assert len(qs)==n
        for model in ('baseline','v03','v05'):
            rows=read_json(ROOT/f'{model}_{label}.json')
            assert len(rows)==n*4
            assert {(r['id'],r['shift']) for r in rows}=={(q,s) for q in qs for s in range(4)}
            for r in rows:
                q=qs[r['id']]; choices=q['choices'][r['shift']:]+q['choices'][:r['shift']]
                answer=q['choices']['ABCD'.index(q['answer_letter'])]
                expected='ABCD'[choices.index(answer)]
                assert r['predicted']==parse_answer(r['raw']) and r['expected']==expected
                assert r['correct']==(r['predicted']==expected)
                idx=q['choices'].index(choices['ABCD'.index(r['predicted'])]) if r['predicted'] in ('A','B','C','D') else None
                assert r['original_choice_index']==idx
            assert sum(r['correct'] for r in rows)==export['summary']['models'][model][label]['overall']['correct']
            export[f'{model}_{label}']=rows
    reference=read_json(Path(__file__).resolve().parent.parent/'docs'/'STATS_ROTATION_V1_RESULTS.json')
    agreement={}
    for name,prior in (('baseline','baseline'),('v03','distilled')):
        old={(r['id'],r['shift']):r['raw'] for r in reference[prior]}
        agreement[name]=sum(old[(r['id'],r['shift'])]==r['raw'] for r in export[name+'_old'])
    import torch
    from safetensors import safe_open
    with safe_open(ROOT/'adapter'/'adapter_model.safetensors',framework='pt',device='cpu') as f:
        keys=list(f.keys()); assert keys
        assert all(torch.isfinite(f.get_tensor(k)).all().item() for k in keys)
    archive=ROOT.parent/(ROOT.name+'.zip')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        manifest=json.loads(z.read('manifest.json'))
        for row in manifest:
            b=z.read(row['path'])
            assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
    export['verification']=dict(manifest_files=len(manifest),finite_tensors=len(keys),
        previous_raw_matches_out_of_240=agreement,archive_bytes=archive.stat().st_size,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    print('V05 VERIFIED',json.dumps(export['verification']),flush=True)
    print('V05 EXPORT',json.dumps(export,ensure_ascii=False),flush=True)


if __name__=='__main__':
    import sys
    corpus() if '--corpus' in sys.argv else verify()
