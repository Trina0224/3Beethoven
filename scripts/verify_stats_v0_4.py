"""Read-only archive, adapter, and independently reconstructed answer checks."""
import hashlib
import json
import zipfile
from pathlib import Path
from stats_v0_3_common import parse_answer


def verify(root):
    archive=root.parent/(root.name+'.zip')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        manifest=json.loads(z.read('manifest.json'))
        for row in manifest:
            b=z.read(row['path'])
            assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
    data={'summary':json.loads((root/'summary.json').read_text())}
    for label,n in (('new',24),('old',60)):
        questions=json.loads((root/f'{label}_benchmark.json').read_text())
        qmap={r['id']:r for r in questions}
        assert len(qmap)==n
        for model in ('baseline','v03','v04'):
            key=f'{model}_{label}'
            rows=json.loads((root/f'{key}.json').read_text())
            assert len(rows)==4*n
            assert {(r['id'],r['shift']) for r in rows}=={(q,s) for q in qmap for s in range(4)}
            for r in rows:
                q=qmap[r['id']]
                original_answer=q['choices']['ABCD'.index(q['answer_letter'])]
                rotated=q['choices'][r['shift']:]+q['choices'][:r['shift']]
                expected='ABCD'[rotated.index(original_answer)]
                assert r['expected']==expected and r['predicted']==parse_answer(r['raw'])
                assert r['correct']==(r['predicted']==expected)
                mapped=q['choices'].index(rotated['ABCD'.index(r['predicted'])]) if r['predicted'] in 'ABCD' else None
                assert r['original_choice_index']==mapped
            assert sum(r['correct'] for r in rows)==data['summary']['models'][model][label]['overall']['correct']
            data[key]=rows
    prior_path=Path(__file__).resolve().parent.parent/'docs'/'STATS_ROTATION_V1_RESULTS.json'
    prior=json.loads(prior_path.read_text())
    agreement={}
    for model,oldname in (('baseline','baseline'),('v03','distilled')):
        before={(r['id'],r['shift']):r['raw'] for r in prior[oldname]}
        agreement[model]=sum(r['raw']==before[(r['id'],r['shift'])] for r in data[f'{model}_old'])
    from safetensors import safe_open
    import torch
    with safe_open(root/'adapter'/'adapter_model.safetensors',framework='pt',device='cpu') as f:
        keys=list(f.keys())
        assert keys and all(torch.isfinite(f.get_tensor(k)).all().item() for k in keys)
    data['verification']=dict(manifest_files=len(manifest),finite_adapter_tensors=len(keys),
        prior_rotation_raw_matches_out_of_240=agreement,archive_bytes=archive.stat().st_size,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    print('V04 VERIFIED',json.dumps(data['verification']),flush=True)
    print('V04 EXPORT',json.dumps(data,ensure_ascii=False),flush=True)
    return data


if __name__=='__main__': verify(Path('/kaggle/working/3beethoven_stats_v0_4'))
