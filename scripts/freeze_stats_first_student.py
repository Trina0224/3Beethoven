"""Freeze only individually verified rows for the limited first student batch.

Does not call a teacher or claim that the full-corpus gate passed.
"""
import argparse
from pathlib import Path
from collections import Counter
from flight_run_stats_v0_3 import read_json, save_json
from stats_consolidation_pilot import build, digest
from stats_consolidation_grader import score, grader_fingerprint
from run_stats_consolidation_compare import validate_verified_rows, training_rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    decision=read_json(repo/'docs/STATS_FIRST_STUDENT_DECISION.json')
    source=read_json(args.source/'verified_teacher_rows.json')
    stories,requests,_,_=build()
    lookup={q['id']:q for s in stories for q in s['questions']}
    verified={split:[] for split in ('train','validation')}
    excluded=[]
    for split,rows in source.items():
        for row in rows:
            judged=score('Expression: '+row['teacher_raw'],lookup[row['source_id']])
            if not judged['primary_correct']:
                excluded.append(row['source_id']);continue
            row=dict(row)
            rec=read_json(args.source/'records'/(row['story_id']+'.json'))
            row['rule_assisted']=not str(rec['accepted'][row['source_id']]['attempt']).startswith('legacy_')
            row['validation']=judged
            verified[split].append(row)
    validate_verified_rows(verified,stories)
    assert len(verified['train'])>=60
    ids={r['source_id'] for r in verified['train']}
    represented={s['archetype'] for s in stories if any(q['id'] in ids for q in s['questions'])}
    assert represented=={s['archetype'] for s in stories}
    args.output.mkdir(parents=True,exist_ok=True)
    gate=dict(passed=True,scope='filtered_pilot_student',full_teacher_gate_passed=False,
              grader_fingerprint=grader_fingerprint(),request_sha256=digest(requests),
              decision_sha256=digest(decision),verified_rows_sha256=digest(verified),
              quality='Every included row passes exact mathematics and reviewed structure; rejected and unresolved rows excluded',
              original_teacher_gate=read_json(args.source/'pilot_original_gate.json'),
              expanded_teacher_gate=read_json(args.source/'pilot_gate.json'),excluded=excluded,
              counts={k:len(v) for k,v in verified.items()},
              train_categories=dict(Counter(r['category'] for r in verified['train'])),
              represented_archetypes=sorted(represented),promotion_eligible=False)
    old=read_json(args.output/'gate.json')
    if old and old!=gate: raise RuntimeError('Frozen pilot-student corpus differs')
    save_json(args.output/'verified_teacher_rows.json',verified)
    save_json(args.output/'gate.json',gate)
    print('FILTERED_STUDENT_FROZEN',gate['counts'],'replay',192,'total',len(training_rows(verified,2027)),flush=True)


if __name__=='__main__': main()
