"""Verify two completed diagnostic runs and make a small portable download."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from flight_run_stats_v0_3 import read_json, save_json, package
from run_stats_consolidation_compare import suite_metrics, gate_result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    root=args.root
    baseline=read_json(root/'v15_selection_baseline.json')
    config=read_json(Path(__file__).resolve().parents[1]/'docs/STATS_FIRST_STUDENT_DECISION.json')
    result=dict(status='two_seed_filtered_pilot_completed_no_promotion',promotion_eligible=False,
                teacher_scope='filtered pilot; full teacher gate did not pass',
                new_training_rows=67,replay_rows=192,baseline=baseline,runs={})
    portable=root.parent/'3beethoven_first_students_download'
    portable.mkdir(exist_ok=True)
    for seed in (2027,31415):
        folder=root/f'original_v15_continued_lora_seed_{seed}'
        complete=read_json(folder/'training_complete.json')
        history=read_json(folder/'validation_history.json')
        selection=read_json(folder/'selection.json')
        manifest=read_json(folder/'weight_manifest.json')
        if not all((complete,history,selection,manifest)):
            raise RuntimeError(f'Incomplete seed {seed}')
        if complete['selection'] != selection or selection['stop_reason']=='manual_review_required':
            raise RuntimeError('Provisional completion is not a completed first batch')
        for record in history:
            metrics={suite:suite_metrics(read_json(folder/f"step_{record['step']}"/(suite+'.json')))
                     for suite in ('old','chain','event')}
            if any(m['pending'] for m in metrics.values()):
                raise RuntimeError('Resolve pending rows before collecting final results')
            gate=gate_result(metrics,baseline,config['validation_gate'])
            if metrics!=record['metrics'] or gate!=record['gate']:
                raise RuntimeError('History disagrees with verified per-question rows')
        first_pass=next((r['step'] for r in history if r['gate']['passed']),None)
        if selection['selected_step']!=first_pass:
            raise RuntimeError('Selection is not the first passing checkpoint')
        step=first_pass if first_pass is not None else selection['diagnostic_step']
        if step is None or (first_pass is None and step!=history[-1]['step']):
            raise RuntimeError('Missing final diagnostic checkpoint')
        if first_pass is None and complete['global_steps'] != complete['actual_max_updates']:
            raise RuntimeError('Nonpassing run has not reached its one-pass cap')
        if first_pass is not None and complete['global_steps'] != first_pass:
            raise RuntimeError('Training continued past the first passing checkpoint')
        if manifest['selected_or_diagnostic_step'] != step:
            raise RuntimeError('Weight manifest refers to a different checkpoint')
        weights=folder/f'step_{step}'/'adapter'/'adapter_model.safetensors'
        sha=hashlib.sha256(weights.read_bytes()).hexdigest()
        if sha!=manifest['adapter_sha256']:
            raise RuntimeError('Portable weights hash mismatch')
        import torch
        import safetensors.torch
        tensors=safetensors.torch.load_file(str(weights))
        if not all(torch.isfinite(t).all() for t in tensors.values()):
            raise RuntimeError('Non-finite portable weights')
        destination=portable/f'seed_{seed}'
        shutil.copytree(weights.parent,destination/'adapter',dirs_exist_ok=True)
        for filename in ('contract.json','training_order.json','training_complete.json','validation_history.json',
                         'selection.json','weight_manifest.json'):
            shutil.copy2(folder/filename,destination/filename)
        for suite in ('old','chain','event'):
            shutil.copy2(folder/f'step_{step}'/(suite+'.json'),destination/(suite+'.json'))
        selected=next(r for r in history if r['step']==step)
        result['runs'][str(seed)]=dict(steps=complete['global_steps'],selection=selection,
             final_invocation_training_loss=complete.get('training_loss'),metrics=selected['metrics'],gate=selected['gate'],
             history=history,weights=manifest)
    shutil.copytree(root/'teacher',portable/'teacher',dirs_exist_ok=True)
    shutil.copytree(root/'v15_selection_rows',portable/'v15_selection_rows',dirs_exist_ok=True)
    if (root/'semantic_reviews').exists():
        shutil.copytree(root/'semantic_reviews',portable/'semantic_reviews',dirs_exist_ok=True)
    for filename in ('environment.json','source_commit.txt','v15_selection_baseline.json'):
        shutil.copy2(root/filename,portable/filename)
    for logfile in root.glob('seed_*.log'):
        shutil.copy2(logfile,portable/logfile.name)
    teacher_usage=root.parent/'3beethoven_first_batch_teacher'/'usage.json'
    if teacher_usage.exists():
        result['teacher_usage']=read_json(teacher_usage)
        shutil.copy2(teacher_usage,portable/'teacher_usage.json')
    save_json(root/'FIRST_STUDENTS_RESULTS.json',result)
    save_json(portable/'FIRST_STUDENTS_RESULTS.json',result)
    text=['第一批兩個 seed 已完成；本批是小批診斷訓練，不升級取代原 v15。',
          '', '67 筆合格新教材＋192 筆 replay；所有待判定項先複核，再決定驗證結果。', '',
          '| 模型 | 使用步數 | 舊技能 /48 | 組合題 /48 | 事件 /16 | 驗證全門檻 |',
          '|---|---:|---:|---:|---:|---|',
          f"| 原 v15 | 0 | {baseline['old']['correct']} | {baseline['chain']['correct']} | {baseline['event']['correct']} | 基準 |"]
    for seed,r in result['runs'].items():
        m=r['metrics'];step=r['selection']['selected_step'] or r['selection']['diagnostic_step']
        text.append(f"| {seed} | {step} | {m['old']['correct']} | {m['chain']['correct']} | {m['event']['correct']} | {'通過' if r['gate']['passed'] else '未通過'} |")
    text += ['', '這些是選點驗證結果，不能當作獨立測試成績。完整教材 gate 未通過，fresh-base 對照及完整四次比較未完成。',
             '', '下載包包含兩個 seed 最早過關（或最後診斷）的完整 adapter、底模 revision、權重 SHA-256、原始驗證回答與訓練資料順序。其他階段權重保存在 Kaggle 完整輸出。']
    report='\n'.join(text)+'\n'
    (root/'FIRST_STUDENTS_REPORT.md').write_text(report)
    (portable/'FIRST_STUDENTS_REPORT.md').write_text(report)
    package(portable)
    print('FIRST_STUDENTS_VERIFIED',json.dumps(result),flush=True)


if __name__=='__main__': main()
