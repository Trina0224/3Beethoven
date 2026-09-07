"""One preregistered lower-LR repair using unchanged data and validation gates.

Run only after CONTROL_COMPLETE and the separate repair decision are frozen.
Previously reviewed answers are reusable only for an identical question/raw hash.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse,sys,json,hashlib,shutil
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--seed',type=int,choices=(2027,31415),required=True);a=ap.parse_args()
    sys.path.insert(0,str(a.repo/'scripts'))
    import run_stats_consolidation_compare as runner
    from stats_consolidation_pilot import digest
    original=a.input/'3beethoven_first_students'
    control=read(a.root/'control/CONTROL_RESULTS.json')
    assert read(a.root/'control/CONTROL_COMPLETE.json')['complete']
    for name in ('v15','step24','step33'):
        assert all(control[name]['suites'][s]['same_raw']==control[name]['suites'][s]['n'] for s in ('old','chain','event')),'Reload control mismatch: diagnose before training'
    decision=read(a.root/'LOW_LR_DECISION.json')
    assert decision['training']['learning_rate']==5e-6
    assert decision['training']['max_updates']==33
    assert decision['repair']['primary_change']=='learning_rate_only'
    output=a.root/'low_lr';output.mkdir(exist_ok=True)
    teacher=output/'teacher';teacher.mkdir(exist_ok=True)
    shutil.copy2(original/'teacher/verified_teacher_rows.json',teacher/'verified_teacher_rows.json')
    gate=read(original/'teacher/gate.json');gate['decision_sha256']=digest(decision)
    save(teacher/'gate.json',gate)
    shutil.copy2(original/'v15_selection_baseline.json',output/'v15_selection_baseline.json')
    shutil.copytree(original/'v15_selection_rows',output/'v15_selection_rows',dirs_exist_ok=True)
    reviews={}
    for suite in ('old','chain','event'):
        for r in read(original/'v15_selection_rows'/f'{suite}.json'):
            if not r.get('review_required',False):reviews[(r['id'],r['raw_sha256'])]=r
        for seed in (2027,31415):
            for step in (12,24,33):
                for r in read(original/f'original_v15_continued_lora_seed_{seed}/step_{step}/{suite}.json'):
                    if not r.get('review_required',False):reviews[(r['id'],r['raw_sha256'])]=r
    original_evaluate=runner.evaluate
    def reviewed_evaluate(model,tokenizer,questions,path,**kwargs):
        rows=original_evaluate(model,tokenizer,questions,path,**kwargs)
        changed=False
        for r in rows:
            prior=reviews.get((r['id'],r['raw_sha256']))
            if r.get('review_required') and prior and prior['question_sha256']==r['question_sha256']:
                if not changed:save(Path(path).with_name(Path(path).stem+'_automatic.json'),rows)
                for key in ('math_correct','review_required','primary_correct','correct','reason'):
                    r[key]=prior[key]
                r['semantic_review_reused']={'source':'Version53 same question and exact raw hash','raw_sha256':prior['raw_sha256']}
                changed=True
        if changed:save(path,rows)
        return rows
    runner.evaluate=reviewed_evaluate
    # Observe actual training microbatches, without changing loss or optimizer.
    from transformers import Trainer
    original_training_step=Trainer.training_step
    trace=output/f'microbatches_{a.seed}.jsonl'
    def traced_step(self,model,inputs,*args,**kwargs):
        item={'global_step_before':self.state.global_step,
              'input_ids_sha256':digest(inputs['input_ids'].detach().cpu().tolist()),
              'labels_sha256':digest(inputs['labels'].detach().cpu().tolist()),
              'supervised_tokens':int((inputs['labels']!=-100).sum().item())}
        loss=original_training_step(self,model,inputs,*args,**kwargs)
        with trace.open('a') as stream:stream.write(json.dumps(item)+'\n')
        return loss
    Trainer.training_step=traced_step
    save(output/f'execution_{a.seed}.json',{'source_runner_sha256':sha(a.repo/'scripts/run_stats_consolidation_compare.py'),'decision_sha256':digest(decision),'seed':a.seed,'review_reuse':'same question SHA and raw SHA only','optimization_change':'LR 2e-5 to 5e-6; all data and gates unchanged'})
    sys.argv=['run_stats_consolidation_compare.py','--start','original_v15_continued_lora','--seed',str(a.seed),'--decision',str(a.root/'LOW_LR_DECISION.json'),'--teacher-rows',str(teacher/'verified_teacher_rows.json'),'--teacher-gate',str(teacher/'gate.json'),'--v15-root',str(a.input/'first_batch_parent'),'--output-root',str(output),'--pilot-student']
    runner.main()
if __name__=='__main__':main()
