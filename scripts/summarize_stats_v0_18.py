"""Produce a reviewable report without changing frozen scores or decisions."""
from run_stats_v0_18 import ROOT
from flight_run_stats_v0_3 import read_json as read

def main():
    result=read(ROOT/'verified_results.json');s=result['summary'];out=result['outputs']
    review=read(ROOT/'semantic_review.json');automatic={n:m['correct'] for n,m in s['tests'].items()}
    if review:s['tests']={n:review['metrics'][n+'_test.json'] for n in s['tests']}
    lines=['# v0.18 concept-depth curriculum: completed results','',
           'The user-requested 1 → 2 → 3 curriculum retains simpler problems and preserves stage checkpoints. Explicit response-level semantic credits, when present, supplement frozen scores; original decisions remain unchanged. No model is promoted on this narrow benchmark.','',
           '## Same untouched 96-question test','',
           '| Candidate | Depth 1 /32 | Depth 2 /32 | Depth 3 /32 | Total /96 | Frozen auto /96 | Unresolved |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,m in s['tests'].items():
        lines.append('| '+name+' | '+' | '.join(str(m['by_depth'][str(d)]['correct']) for d in (1,2,3))+f" | {m['correct']} | {automatic[name]} | {m['pending']} |")
    lines+=['','The baseline already scores 16/16 on depth-1 validation in this run. Do not claim those primitive skills first emerged during stage 1; the stage trajectory tests retention and subsequent composition.','',
            '## Where capabilities appear','',
            '| Track and depth /8 | '+' | '.join(s['tests'])+' |',
            '|---|'+ '|'.join('---:' for _ in s['tests'])+'|']
    for cell in s['tests']['v15']['by_cell']:
        lines.append('| '+cell+' | '+' | '.join(str(m['by_cell'][cell]['correct']) for m in s['tests'].values())+' |')
    lines+=['','## Validation stopping and retained earlier skills','',
            '| Stage | Epochs | Updates | Stop reason | Mastery | Depth 1 /16 | Depth 2 /16 | Depth 3 /16 |',
            '|---|---:|---:|---|---|---:|---:|---:|']
    for st in s['stages']:
        lines.append(f"| {st['stage']} | {st['epochs']} | {st['steps']} | {st['stop_reason']} | {st['mastered']} | "+' | '.join(str(st['validation']['by_depth'][str(d)]['correct']) for d in (1,2,3))+' |')
    lines+=['','A low plateau is not mastery. An eight-epoch cap without convergence stops progression; it is not silently labeled convergence. All endpoint choices and the control budget were fixed before final-test answers. Full epoch validation histories and raw answers are preserved.','',
            '## Matched shuffled exposure','']
    final=f"stage_{len(s['stages'])}";control=f"control_{len(s['stages'])}"
    a={r['id']:r for r in out[control+'_test.json']};b={r['id']:r for r in out[final+'_test.json']}
    wins=sum(b[i]['correct'] and not a[i]['correct'] for i in a);losses=sum(a[i]['correct'] and not b[i]['correct'] for i in a)
    if review:wins,losses=(review['curriculum_vs_control'][k] for k in ('newly_correct','newly_wrong'))
    lines += [f"Against the shuffled control, the final curriculum newly answers {wins} questions correctly and newly misses {losses}, for a net {wins-losses:+d}/96. Both arms see the exact same {result['verification']['matched_training_exposures']} training-row exposures, with matching optimizer updates and reset boundaries.",'',
              'The control globally shuffles the realized multiset. Its budget is determined by curriculum validation, not independently optimized for the control. One seed and adaptive stopping limit causal/general claims. A small difference does not establish a reliable universal order effect.','',
              '## Scope and reproducibility','',
              'New targets are exact procedural supervised references, not newly generated Llama teacher responses. Teacher calls and API cost are zero. Both arms start from the hash-verified v15 adapter and pinned original base. The constant LR is 2e-5 and effective batch is 8. This differs from v17 in both data and LR; cross-run improvement cannot be attributed solely to order.','',
              'Concept depth counts explicitly listed domain-rule applications, not internal reasoning steps. Primitive one-step Poisson-variance questions necessarily have a scalar reference, so their score alone does not establish robust reasoning. Other tasks require verified expression structure as well as exact arithmetic agreement.','',
              'Question stories and primitive identities within tracks are split before training; previous overlapping full-task identities are blocked. Fresh parameters in known chain templates do not measure unseen concepts. Affine second moments and broad retention on the old eight-family benchmark are not established by this run.','',
              f"Verified {result['verification']['responses']} generated responses, {len(result['verification']['stage_checkpoints'])} full curriculum epoch checkpoints, and {len(result['verification']['weights'])} boundary adapters. Truncated responses: {result['verification']['truncated']}. Initially pending response-level reviews across validation and test: {len(result['pending_semantic_review'])}; explicit semantic credits: {len(review['credits']) if review else 0}.",'',
              'Two baseline depth-2 validation responses correctly factor the Poisson second moment as mean*(mean+1). The supplemental baseline depth-2 count is 14/16, versus frozen automatic 12/16. This correction does not alter any stage transition.','',
              'Full optimizer/RNG checkpoints are preserved in Kaggle output. The compact ZIP contains stage/control boundary adapters, data, source, histories and results; it excludes full optimizer checkpoints. Final archive hash and saved Kaggle version are recorded in MODEL_BACKUP_STATUS.json. A fresh-session restoration of new v18 weights is not claimed.','']
    transfer=read(ROOT/'transfer_results.json')
    if transfer:
        lines+=['## Historical eight-family transfer diagnostic','',transfer['protocol'],'',
                '| Candidate | Automatic /96 | Pending | Affine second moment /12 | Scaled Poisson variance /12 |',
                '|---|---:|---:|---:|---:|']
        for name,m in transfer['metrics'].items():
            lines.append(f"| {name} | {m['correct']} | {m['pending']} | {m['by_category']['moment']} | {m['by_category']['poisson_scaled']} |")
        lines+=['','These are the same historical questions and prompt as v17. The 96 v15 responses were revalidated and reused; 192 responses from final curriculum/control were newly generated. They are additional to the primary experiment response count above. This diagnostic does not affect model selection. Raw per-family results are in STATS_V0_18_TRANSFER_RESULTS.json. Old MC retention is not remeasured.','']
    (ROOT/'report.md').write_text('\n'.join(lines))
    print('V18 REPORT WRITTEN',flush=True)

if __name__=='__main__':main()
