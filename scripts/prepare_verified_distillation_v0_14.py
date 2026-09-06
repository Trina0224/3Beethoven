"""Create teacher-derived targets without copying reference formulas."""
import json
from pathlib import Path
from formulation_grader import grade
from stats_curriculum_v0_13 import digest


def formulation_prompt(q):
    return q['question']+'\nReturn one line: Expression: <a fully substituted numerical expression>. Keep arithmetic and unit conversion operations unevaluated. Use integers, fractions, +, -, *, /, ** and comb(n,r). No final answer or explanation.'


def build(repo):
    data=json.loads((repo/'docs/STATS_V0_14_FROZEN_QUESTIONS.json').read_text())
    raw=json.loads((repo/'docs/STATS_V0_14_TEACHER_RAW.json').read_text())
    output={}
    for split in ('train','validation'):
        lookup={q['id']:q for q in data[split]};rows=[]
        for record in raw[split]:
            q=lookup[record['id']]
            for attempt in record['attempts']:
                judged=grade(attempt['raw'],q)
                if judged['math_correct'] is True:
                    rows.append(dict(source_id=q['id'],question_sha256=digest(q),prompt=formulation_prompt(q),
                        target='Expression: '+judged['normalized_expression'],teacher_raw=attempt['raw'],
                        selected_attempt=attempt['attempt'],normalization='AST expansion of teacher-owned assignments and decimal normalization; no reference bindings inserted'))
                    break
        output[split]=rows
    return output


if __name__=='__main__':
    repo=Path(__file__).resolve().parents[1]
    data=build(repo)
    (repo/'docs/STATS_V0_14_VERIFIED_DISTILLATION.json').write_text(json.dumps(data,indent=2)+'\n')
    print({s:len(rows) for s,rows in data.items()})
