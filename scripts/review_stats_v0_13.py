"""Explicit retrospective annotations, not a general semantic grading model."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
# Credit a complete correct setup explicitly present in raw output. Later
# contradictory setups remain flagged; this is not tool-ready final accuracy.
PASSES={
 'baseline':[5,6,10]+list(range(12,24))+[39,61,74],
 'v10':[0,2,3,4,5,7,10]+[12,13,14,15,16,18,19,20,21,22,23]+
       [36,37,38,39,41,42,45]+[48,49,51,52,54,55,59]+[60,61,67,68,70]+[74,81],
 'v13':list(range(96)),
}
CONFLICTS={'baseline':[5], 'v10':[4,12,13,14,15,19,20,22,23,48,49,52], 'v13':[]}
AMBIGUOUS={'baseline':[], 'v10':[69,75,79], 'v13':[]}


def main():
    import hashlib
    path=ROOT/'docs/STATS_V0_13_RESULTS.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='4e2123af552206f6c76651449ed8f8baa308987b038dca49a1a4954d66a72ca4', 'Source changed; annotations need fresh review'
    data=json.loads(path.read_text());qs=json.loads((ROOT/'docs/STATS_V0_13_FROZEN_QUESTIONS.json').read_text())['test']
    ledger=[];summary={}
    for name,rows in data.items():
        for i,row in enumerate(rows):
            assert row['id']==qs[i]['id']
            reason=('A complete correct setup is explicitly present; subsequent arithmetic excluded.' if i in PASSES[name]
                    else 'No complete correct problem-specific setup identified; scalar answers alone are insufficient.')
            if i in CONFLICTS[name]:reason+=' Later output introduces a conflicting setup; presence credit does not certify a usable final response.'
            if i in AMBIGUOUS[name]:reason='Numeric total can be right, but event/probability bindings are mislabeled. Excluded from confirmed credit.'
            ledger.append(dict(model=name,index=i,id=row['id'],category=row['category'],question=qs[i]['question'],raw=row['raw'],
                correct_setup_present=i in PASSES[name],conflicting_setup=i in CONFLICTS[name],ambiguous_binding=i in AMBIGUOUS[name],reason=reason))
        summary[name]=dict(n=96,correct_setup_present=len(PASSES[name]),conflicting_among_present=len(CONFLICTS[name]),ambiguous_binding=len(AMBIGUOUS[name]))
    out=dict(method='Non-blinded retrospective explicit-output review. Presence of correct setup, not final-tool-interface accuracy. All 192 baseline/v10 answers read; v13 also checked by exact reference structure in the original scorer.',
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),summary=summary,records=ledger)
    (ROOT/'docs/STATS_V0_13_FORMAT_INDEPENDENT_REVIEW.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
