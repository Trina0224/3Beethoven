"""Preserve full evidence and portable fixed adapters after multiview evaluation."""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root;repo=Path(__file__).resolve().parents[1]
    summary=read(root/'MULTIVIEW_SUMMARY.json')
    assert not summary['pending']
    assert set(summary['probe'])=={'v15','2027','31415'}
    assert all(v['n']==48 for v in summary['probe'].values())
    import torch
    from safetensors.torch import load_file
    weights={}
    for seed in (2027,31415):
        folder=root/f'seed_{seed}'
        complete=read(folder/'training_complete.json');selection=read(folder/'selection.json')
        assert complete['global_steps']==65 and complete['actual_microbatches_verified']==516
        for step in (12,24,48,65):
            path=folder/f'step_{step}/adapter/adapter_model.safetensors'
            tensors=load_file(str(path));assert len(tensors)==392
            assert all(torch.isfinite(t).all() for t in tensors.values())
            weights[f'{seed}/step_{step}']={'sha256':sha(path),'bytes':path.stat().st_size,'finite_tensors':len(tensors)}
        portable=root/'portable'/str(seed)
        shutil.copytree(folder/f'step_{selection["fixed_step"]}/adapter',portable,dirs_exist_ok=True)
    parent=a.input/'first_batch_parent/3beethoven_stats_v0_15/adapter'
    assert sha(parent/'adapter_model.safetensors')=='9369d52de4a886df9da0c872cd41bd4e01af0a38bf02ad724b5951c1a6b9f5d3'
    shutil.copytree(parent,root/'portable/v15_parent',dirs_exist_ok=True)
    (root/'WEIGHT_MANIFEST.json').write_text(json.dumps(weights,indent=2)+'\n')
    source=root/'executed_source'
    for name in ('scripts','docs','tests'):
        for src in (repo/name).rglob('*'):
            if not src.is_file() or src.suffix not in ('.py','.json','.md'):continue
            dst=source/src.relative_to(repo);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    files=[]
    for src in root.rglob('*'):
        if not src.is_file() or src==a.output:continue
        rel=src.relative_to(root)
        if 'checkpoints' in rel.parts:continue
        if rel.parts[0].startswith('seed_') and 'adapter' in rel.parts:continue
        if src.suffix not in ('.json','.jsonl','.md','.py','.log','.safetensors','.model','.txt'):continue
        files.append(src)
    manifest={str(src.relative_to(root)):{'sha256':sha(src),'bytes':src.stat().st_size} for src in sorted(files)}
    with zipfile.ZipFile(a.output,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for src in sorted(files):z.write(src,str(src.relative_to(root)))
        z.writestr('ARCHIVE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(a.output) as z:assert z.testzip() is None
    receipt={'file':a.output.name,'bytes':a.output.stat().st_size,'sha256':sha(a.output),
        'manifest_files':len(manifest),'portable_models':['2027','31415','v15_parent'],
        'full_checkpoint_weights':'retained separately in saved Kaggle output',
        'teacher_calls':0,'optimizer_updates':130,'verified_microbatches':1032}
    a.output.with_suffix('.manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('MULTIVIEW_PACKAGE_COMPLETE',json.dumps(receipt),flush=True)

if __name__=='__main__':main()
