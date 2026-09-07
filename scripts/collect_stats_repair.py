"""Collect immutable evidence from the completed bounded repair and controls."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    a = p.parse_args()
    root = a.root
    summary = read(root / 'REPAIR_SUMMARY.json')
    assert not summary['pending']
    prepared = read(root / 'prepared_parent/RESULTS.json')
    assert read(root / 'prepared_parent/COMPLETE.json')['complete']
    assert not prepared['pending']
    traces = read(root / 'ACTUAL_TRACE_VERIFIED.json')
    result = {'experiment': 'bounded_low_lr_response_distillation_repair',
              'teacher_calls': 0, 'independent_test_executed': False,
              'original_gates_unchanged': True, 'controls': summary['controls'],
              'prepared_parent_control': prepared, 'actual_training_trace': traces,
              'repairs': summary['repairs'], 'original_lr_runs': {},
              'decision': read(root / 'LOW_LR_DECISION.json')}
    for seed in (2027, 31415):
        name = f'original_v15_continued_lora_seed_{seed}'
        folder = root / 'low_lr' / name
        completed = result['repairs'][str(seed)]['complete']
        assert completed and completed['selection']['stop_reason'] != 'manual_review_required'
        assert traces[str(seed)]['matched_all_actual_microbatches']
        result['repairs'][str(seed)]['weight_manifest'] = read(folder / 'weight_manifest.json')
        result['repairs'][str(seed)]['execution'] = read(root / f'low_lr/execution_{seed}.json')
        result['original_lr_runs'][seed] = read(a.input / '3beethoven_first_students' / name / 'validation_history.json')
    result['total_optimizer_updates'] = sum(t['updates'] for t in traces.values())
    result['passing_repairs'] = [seed for seed, r in result['repairs'].items()
                                 if r['complete']['selection']['selected_step'] is not None]
    result['promotion_decision'] = ('retain_original_v15_no_repair_passed' if not result['passing_repairs']
                                    else 'selection_pass_exists_independent_test_still_required')
    (root / 'STATS_REPAIR_RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
    print('REPAIR_COLLECTION_COMPLETE', json.dumps({'updates': result['total_optimizer_updates'],
          'passing_repairs': result['passing_repairs'], 'promotion_decision': result['promotion_decision']}), flush=True)


if __name__ == '__main__':
    main()
