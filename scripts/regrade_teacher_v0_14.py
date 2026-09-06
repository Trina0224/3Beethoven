"""Regrade saved attempts without any API requests or changes to raw data."""
import json
from pathlib import Path
from collections import Counter
from formulation_grader import grade

ROOT=Path(__file__).resolve().parents[1]/'docs'


def main():
    data=json.loads((ROOT/'STATS_V0_14_TEACHER_RAW.json').read_text())
    questions=json.loads((ROOT/'STATS_V0_14_FROZEN_QUESTIONS.json').read_text())
    lookup={q['id']:q for rs in questions.values() for q in rs}
    out={};summary={}
    for split,rows in data.items():
        out[split]=[]
        for r in rows:
            attempts=[dict(attempt=a['attempt'],raw=a['raw'],old_judged=a['judged'],new_judged=grade(a['raw'],lookup[r['id']])) for a in r['attempts']]
            out[split].append(dict(id=r['id'],category=r['category'],attempts=attempts,
                any_verified=any(a['new_judged']['math_correct'] is True for a in attempts)))
        summary[split]=dict(n=len(rows),first_attempt_verified=sum(r['attempts'][0]['new_judged']['math_correct'] is True for r in out[split]),
            any_attempt_verified=sum(r['any_verified'] for r in out[split]),
            attempts=dict(Counter('verified' if a['new_judged']['math_correct'] is True else 'incorrect' if a['new_judged']['math_correct'] is False else 'review' for r in out[split] for a in r['attempts'])))
    result=dict(method='Representation-tolerant exact execution plus reviewed formula structures. Unrecognized equivalence stays pending, not wrong. Reference values never fill model bindings.',summary=summary,records=out,new_api_calls=0)
    (ROOT/'STATS_V0_14_TEACHER_REGRADED.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
