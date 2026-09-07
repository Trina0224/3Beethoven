"""Zero-call transport/semantic audit. Frozen paid-run gates are never rewritten."""
import ast
import json
from collections import Counter
from fractions import Fraction as F
from stats_teacher_compact import DOCS, DATA, judge, payload, read_json, save_json, digest, reviewed_shape, spec_for
from stats_teacher_envelope import _unique_object
from formulation_grader import parse_expression


def audit_response(raw,q,finish):
    if finish!='stop':
        return dict(accepted=False,classification='truncated_output',repairs=[])
    repairs=[]
    decoder=json.JSONDecoder(object_pairs_hook=_unique_object)
    try:
        stripped=raw.strip()
        if stripped.startswith('{'):
            obj,end=decoder.raw_decode(stripped)
            suffix=stripped[end:].strip()
            if suffix:
                if suffix!='}':
                    raise ValueError('Unrecognized suffix')
                repairs.append('removed_one_redundant_closing_brace')
        else:
            obj,repair=payload(raw)
            if repair:
                repairs.append(repair)
        if isinstance(obj,dict) and 'output_structure' in obj:
            if set(obj)!={'question_id','question','output_structure'} or obj['question_id']!=q['id'] or obj['question']!=q['question']:
                raise ValueError('Ambiguous or mismatched echoed request')
            obj=obj['output_structure']
            repairs.append('extracted_verified_echoed_output_structure')
        if not isinstance(obj,dict) or set(obj)!={'intermediates','answers'}:
            raise ValueError('Invalid answer envelope')
        # Match the exact bounded representation normalizer already used on final expressions.
        original_steps=dict(obj['intermediates'])
        obj['intermediates']={k:ast.unparse(parse_expression(v,{})) for k,v in original_steps.items()}
        result=judge(json.dumps(obj),q,finish)
        result.update(repairs=repairs,original_intermediates=original_steps)
        if result['classification']=='equivalence_pending' and spec_for(q)['kind']=='interval':
            s=spec_for(q)
            lo,hi,d=(F(s[k]) for k in ('lower','upper','divisor'))
            center,half=(lo+hi)/2,(hi-lo)/2
            def n(x):
                return str(x.numerator) if x.denominator==1 else f'({x.numerator}/{x.denominator})'
            refs=[f'{n(center)}+{n(half)}/{n(d)}',
                  f'({n(lo)}+{n(hi)})/2+({n(hi)}-({n(lo)}+{n(hi)})/2)/{n(d)}']
            if all(result['intermediate_checks'].values()) and any(reviewed_shape(result['expression'])==reviewed_shape(ref) for ref in refs):
                result.update(accepted=True,classification='accepted',semantic_review='Verified fixed midpoint plus reduced half-width; partial arithmetic evaluation only')
        return result
    except (ValueError,TypeError,KeyError,SyntaxError) as exc:
        return dict(accepted=False,classification='unsupported_or_malformed',error=str(exc),repairs=repairs)


def review():
    evidence=read_json(DOCS/'STATS_TEACHER_COMPACT_EVIDENCE.json')
    questions={q['id']:q for q in read_json(DATA)['questions']}
    records={json.loads(r['request']['messages'][1]['content'])['question_id']:r for r in evidence['responses']}
    rows=[]
    for original in evidence['results.json']['rows']:
        q=questions[original['id']]
        r=records[q['id']]
        checked=audit_response(r['text'],q,r['finish_reason'])
        if q['id']=='teacher_ab_interval_04':
            obj=json.loads(r['text'])
            assert obj['intermediates']['new_half_width']=='(223-34)/2/2**0.5*2**0.5/2**0.5*4**0.5'
            checked.update(accepted=False,classification='mathematical_error',semantic_review='The multiplicative factor after old half-width simplifies to sqrt(2), whereas multiplying sample size by 4 requires factor 1/2. The displayed final expression uses the same wrong factor. No irrational completion is invented.')
        elif q['id']=='teacher_ab_interval_02':
            assert '66.5*sqrt(25)' in r['text']
            checked.update(accepted=False,classification='mathematical_error',semantic_review='New half-width is 66.5*5 instead of 66.5/5; multiplying sample size by 25 must shrink, not enlarge, the half-width.')
        rows.append(dict(id=q['id'],family=q['development_family'],frozen_classification=original['classification'],**checked))
    counts=Counter(r['family'] for r in rows if r['accepted'])
    pending=[r['id'] for r in rows if r['classification'] in ('equivalence_pending','unsupported_or_malformed')]
    old=[]
    for r in read_json(DOCS/'STATS_TEACHER_AB_EVIDENCE.json')['responses']:
        if 'intermediates object' in r['request']['messages'][0]['content']:
            q=questions[json.loads(r['request']['messages'][1]['content'])['question_id']]
            old.append(dict(id=q['id'],family=q['development_family'],**audit_response(r['text'],q,r['finish_reason'])))
    old_lookup={r['id']:r for r in old}
    paired=dict(Counter(f"B{int(old_lookup[r['id']]['accepted'])}_C{int(r['accepted'])}" for r in rows))
    return dict(role='posthoc_diagnostic_not_execution_gate',teacher_calls_added=0,source_digest=digest(evidence),
                accepted=sum(counts.values()),n=len(rows),by_family=dict(counts),pending=pending,
                frozen_gate_passed=evidence['results.json']['passed'],training_released=False,
                hypothetical_same_threshold_passed=sum(counts.values())>=22 and all(counts[f]>=7 for f in ('poisson_process','interval','affine_poisson')) and not pending,
                old_B_same_diagnostic_accepted=sum(r['accepted'] for r in old),paired=paired,
                classifications=dict(Counter(r['classification'] for r in rows)),rows=rows)


if __name__=='__main__':
    result=review()
    path=DOCS/'STATS_TEACHER_COMPACT_REVIEW.json'
    if path.exists():
        raise SystemExit('Existing review preserved')
    save_json(path,result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
