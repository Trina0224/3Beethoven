"""Review collection and frozen post-training selection for the expansion pilot."""
import argparse,json,hashlib
from pathlib import Path
from run_stats_consolidation_compare import suite_metrics,gate_result
from stats_consolidation_pilot import digest

REPO=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def save(p,value):Path(p).write_text(json.dumps(value,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--select',action='store_true');p.add_argument('--review-file',type=Path)
    a=p.parse_args();root=a.root
    decision=read(REPO/'docs/STATS_EXPANSION_DECISION.json')
    matrix=read(REPO/'docs/STATS_CONSOLIDATION_SELECTION_VALIDATION.json')['suites']
    questions={q['id']:q for group in matrix.values() for q in group}
    questions.update({q['id']:q for q in read(REPO/'docs/STATS_EXPANSION_CORPUS.json')['transfer_probe']})
    if a.review_file:
        entries=read(a.review_file)
        for review in entries:
            path=root/review['path'];assert path.resolve().is_relative_to(root.resolve())
            rows=read(path);row=next(r for r in rows if r['id']==review['id'])
            assert row['raw_sha256']==review['raw_sha256'] and row['question_sha256']==review['question_sha256']
            assert review['math_correct'] in (True,False) and review['reason']
            assert row['question_sha256']==digest(questions[row['id']])
            row.update(math_correct=review['math_correct'],review_required=False,
                primary_correct=bool(review['math_correct'] and row['executable']),
                correct=bool(review['math_correct'] and row['executable']),reason=review['reason'],
                semantic_review={'review_file_sha256':sha(a.review_file),'reason':review['reason']})
            save(path,rows)
    pending=[];summary={'baseline':read(root/'v15_selection_baseline.json'),'seeds':{},'probe':{}}
    paths=list((root/'v15_selection_rows').glob('*.json'))
    for seed in (2027,31415):
        folder=root/f'seed_{seed}';history=[]
        complete=read(folder/'training_complete.json') if (folder/'training_complete.json').exists() else None
        for step in decision['validation_steps']:
            stepdir=folder/f'step_{step}'
            if not all((stepdir/f'{suite}.json').exists() for suite in matrix):continue
            metrics={suite:suite_metrics(read(stepdir/f'{suite}.json')) for suite in matrix}
            history.append({'step':step,'metrics':metrics,'gate':gate_result(metrics,summary['baseline'],decision['validation_gate'])})
            paths.extend(stepdir/f'{suite}.json' for suite in matrix)
        summary['seeds'][str(seed)]={'training_complete':complete,'history':history}
    paths.extend((root/'probe').glob('*/rows.json'))
    for path in paths:
        if '_automatic' in path.name:continue
        for row in read(path):
            if row.get('review_required'):
                pending.append({'path':str(path.relative_to(root)),'id':row['id'],
                    'question_sha256':row['question_sha256'],'raw_sha256':row['raw_sha256'],
                    'raw':row['raw'],'computed':row.get('computed'),'reason':row.get('reason'),
                    'question':questions[row['id']]})
    summary['pending']=pending
    if a.select:
        assert not pending,'Resolve every pending response before selection'
        assert all(v['training_complete'] and len(v['history'])==4 for v in summary['seeds'].values())
        for seed,item in summary['seeds'].items():
            passing=next((r for r in item['history'] if r['gate']['passed']),None)
            selection={'selected_step':passing['step'] if passing else None,
                'fixed_step':passing['step'] if passing else decision['final_step'],'all_pending_resolved':True,
                'reason':'earliest_full_gate_pass' if passing else 'final_diagnostic_no_full_gate_pass',
                'decision_sha256':digest(decision),'reviewed_history_sha256':digest(item['history'])}
            path=root/f'seed_{seed}/selection.json'
            if path.exists():assert read(path)==selection,'Do not reselect from probe results'
            save(path,selection);save(root/f'seed_{seed}/reviewed_validation_history.json',item['history'])
            item['selection']=selection
    probe_questions=read(REPO/'docs/STATS_EXPANSION_CORPUS.json')['transfer_probe']
    lookup={q['id']:q for q in probe_questions}
    for path in (root/'probe').glob('*/rows.json'):
        rows=read(path);cells={}
        for domain in ('events','moments'):
            for target in sorted({q['semantics']['target'] for q in probe_questions if q['semantics']['kind']==domain}):
                group=[r for r in rows if lookup[r['id']]['semantics']['kind']==domain and lookup[r['id']]['semantics']['target']==target]
                cells[domain+'_'+target]=suite_metrics(group)
        summary['probe'][path.parent.name]={'n':len(rows),'cells':cells,'overall':suite_metrics(rows)}
    save(root/'EXPANSION_SUMMARY.json',summary)
    print('EXPANSION_SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__':main()
