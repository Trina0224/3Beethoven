"""Preserve audit documents and verification source beside the trained adapter."""
import shutil
import subprocess
from pathlib import Path
from flight_run_stats_v0_3 import read_json,save_json,package
from verify_stats_v0_5 import ROOT,verify

def main():
    assert read_json(ROOT/'summary.json'), 'Complete training and evaluation first'
    audit=read_json(ROOT/'content_review.json')
    assert audit['audited_records']==204 and len(audit['revisions'])==27
    repo=Path(__file__).resolve().parent.parent
    (ROOT/'documentation').mkdir(exist_ok=True)
    for name in ('STATS_V0_5_PROTOCOL.md','STATS_V0_5_DATA_AUDIT.md','STATS_V0_5_FROZEN_QUESTIONS.json','TEACHER_POLICY.md'):
        shutil.copy2(repo/'docs'/name,ROOT/'documentation'/name)
    for name in ('verify_stats_v0_5.py','finalize_stats_v0_5.py','test_stats_v0_5.py'):
        shutil.copy2(repo/'scripts'/name,ROOT/'source'/name)
    save_json(ROOT/'preservation.json',dict(
        repository='Trina0224/3Beethoven',
        training_source_commit='63470f9a01f3c40839ee9cae0e92bef70c7af56a',
        finalization_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        prior_corpus_checkpoint=dict(kaggle_version=8,script_version_id=347596444),
        independent_review='Assistant read all 204 initial targets and all 27 final revisions before training; no further target edits after frozen training hashes',
        final_targets='Llama-only, verified against exact cached responses'))
    package(ROOT)
    verify()

if __name__=='__main__': main()
