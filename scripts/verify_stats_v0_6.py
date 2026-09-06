"""Independent response and artifact checks; export only public experiment data."""
import hashlib
import json
import zipfile
from pathlib import Path
from flight_run_stats_v0_3 import read_json,save_json
from stats_curriculum_v0_5 import build
from stats_v0_3_common import digest,parse_answer
from generate_stats_v0_5 import parse_output

ROOT=Path('/kaggle/working/3beethoven_stats_v0_6')


def verify():
    export={'summary':read_json(ROOT/'summary.json'),
            'environment':read_json(ROOT/'environment.json'),
            'trainer_log':read_json(ROOT/'trainer_log.json')}
    for label,n in (('new',48),('old',60)):
        qs={r['id']:r for r in read_json(ROOT/(label+'_benchmark.json'))}
        assert len(qs)==n
        for model in ('baseline','v05','v06'):
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
    reference=read_json(Path(__file__).resolve().parent.parent/'docs'/'STATS_V0_5_RESULTS.json')
    agreement={}
    for name in ('baseline','v05'):
        old={(r['id'],r['shift']):r['raw'] for r in reference[name+'_old']}
        agreement[name]=sum(old[(r['id'],r['shift'])]==r['raw'] for r in export[name+'_old'])
    export['teacher']=read_json(ROOT/'teacher_test.json')
    export['pairing']=read_json(ROOT/'pairing_protocol.json')
    summary=export['summary']['models']
    accuracy=summary['v06']['new']['overall']['accuracy']
    export['goals']=dict(at_least_60_percent=accuracy>=0.6,
        double_baseline=accuracy>=2*summary['baseline']['new']['overall']['accuracy'],
        half_questions_all_rotations_correct=summary['v06']['new']['all_four_correct']>=24)
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
    print('V06 VERIFIED',json.dumps(export['verification']),flush=True)
    print('V06 EXPORT',json.dumps(export,ensure_ascii=False),flush=True)


if __name__=='__main__':
    import sys
    verify()
