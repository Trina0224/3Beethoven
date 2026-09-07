"""Offline audit of the SHA-pinned first-student portable ZIP; no model calls.

Requires tokenizers and jinja2. Uses the saved tokenizer and chat template, then
checks totals against the runtime contract. Never rewrites historical scores.
"""
import argparse
import ast
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

ZIP_SHA = '9090815c7b49f82cfeba0c8b5a95eece1259f9f0eea3c526e4c4d3661f26af36'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def audit(root):
    from tokenizers import Tokenizer
    from jinja2 import Environment
    result = {'source_zip_sha256': ZIP_SHA, 'method': 'retrospective offline audit; no new inference or training', 'runs': {}}
    expected = read(root / 'FIRST_STUDENTS_RESULTS.json')
    for seed in (2027, 31415):
        folder = root / f'seed_{seed}'
        contract = read(folder / 'contract.json')
        order = read(folder / 'training_order.json')
        assert digest(order) == contract['training_rows_sha256']
        assert contract['parent_adapter_sha256'] == '9369d52de4a886df9da0c872cd41bd4e01af0a38bf02ad724b5951c1a6b9f5d3'
        manifest = read(folder / 'weight_manifest.json')
        assert hashlib.sha256((folder / 'adapter/adapter_model.safetensors').read_bytes()).hexdigest() == manifest['adapter_sha256']
        tok = Tokenizer.from_file(str(folder / 'adapter/tokenizer.json'))
        template = Environment().from_string((folder / 'adapter/chat_template.jinja').read_text())
        totals, sources = Counter(), defaultdict(Counter)
        for row in order:
            messages = [{'role': 'user', 'content': row['prompt']}]
            kwargs = {'bos_token': '<|begin_of_text|>', 'date_string': '07 Sep 2026'}
            prefix = tok.encode(template.render(messages=messages, add_generation_prompt=True, **kwargs), add_special_tokens=False).ids
            full = tok.encode(template.render(messages=messages + [{'role': 'assistant', 'content': row['target']}], add_generation_prompt=False, **kwargs), add_special_tokens=False).ids
            assert full[:len(prefix)] == prefix and 0 < len(prefix) < len(full) <= 768
            labels = [-100] * len(prefix) + full[len(prefix):]
            assert tok.decode([v for v in labels if v != -100], skip_special_tokens=False) == row['target'] + '<|eot_id|>'
            totals.update(rows=1, input_tokens=len(full), supervised_tokens=len(full)-len(prefix))
            sources[row['source_group']].update(rows=1, supervised_tokens=len(full)-len(prefix))
        assert dict(totals) == contract['token_accounting']
        run = {'contract_verified': True, 'supervision_verified': dict(totals), 'supervision_by_source': dict(sources), 'segments': [], 'evaluations': {}, 'transitions_24_to_33': {}, 'loss_invocations': []}
        for start, end in ((0,96),(96,192),(192,259)):
            subset = order[start:end]
            run['segments'].append({'planned_rows_1based': [start+1,end], 'source_counts':dict(Counter(r['source_group'] for r in subset)), 'category_counts':dict(Counter(r['category'] for r in subset))})
        for suite in ('old','chain','event'):
            stages = {0:read(root / 'v15_selection_rows' / f'{suite}.json')}
            for step in (12,24,33):
                rows = read(folder / f'{suite}.json') if step == 33 else read(root / f'semantic_reviews/{seed}_step{step}_automatic/{suite}.json')
                reviews = read(root / f'semantic_reviews/step{step}_review.json')['reviews']
                for row in rows:
                    assert hashlib.sha256(row['raw'].encode()).hexdigest() == row['raw_sha256']
                    for rev in reviews:
                        if rev['seed'] == seed and rev['id'] == row['id']:
                            assert rev['raw_sha256'] == row['raw_sha256']
                            row['primary_correct'] = rev['correct']
                stages[step] = rows
                wanted = next(h for h in expected['runs'][str(seed)]['history'] if h['step']==step)['metrics'][suite]['correct']
                assert sum(r['primary_correct'] for r in rows) == wanted
            run['evaluations'][suite] = {}
            for step, rows in stages.items():
                cats=defaultdict(lambda: {'n':0,'correct':0})
                for row in rows:
                    cats[row['category']]['n']+=1;cats[row['category']]['correct']+=int(row['primary_correct'])
                run['evaluations'][suite][step] = {'correct':sum(r['primary_correct'] for r in rows),'n':len(rows),'token_limit':sum(r['hit_token_limit'] for r in rows),'strict_one_line':sum(r['strict_one_line_expression'] for r in rows),'categories':dict(cats)}
            before={r['id']:r for r in stages[24]};flips=[];counts=Counter()
            for after in stages[33]:
                prior=before[after['id']]
                assert prior['prompt']==after['prompt'] and prior['question_sha256']==after['question_sha256']
                key=f"{int(prior['primary_correct'])}->{int(after['primary_correct'])}";counts[key]+=1
                if prior['primary_correct']!=after['primary_correct']:
                    flips.append({'id':after['id'],'category':after['category'],'transition':key,'prompt':after['prompt'],'before':prior['raw'],'after':after['raw'],'reason':after['reason']})
            run['transitions_24_to_33'][suite]={'counts':dict(counts),'flips':flips}
        for name,start,end in (('launch_0',0,12),('resume12',12,24),('resume24',24,33)):
            lines=(root/f'seed_{seed}_{name}.log').read_text().splitlines()
            summary=next(ast.literal_eval(l) for l in lines if l.startswith("{'train_runtime':"))
            raw=float(summary['train_loss']);updates=end-start
            run['loss_invocations'].append({'start':start,'end':end,'raw_rounded_log_loss':raw,'corrected_approx_mean_per_update':raw*end/updates,'note':'Approximate: log values are rounded; independent of promotion scoring.'})
        result['runs'][seed]=run
    result['limitations']=['Saved row order is the planned sampler order, not a per-batch consumption trace.', 'Optimizer/RNG restoration is supported by runner/framework source, but not proven by an uninterrupted-versus-resumed numerical control.', 'Token boundaries are reconstructed from saved tokenizer/template and match runtime totals; no gradient hooks were recorded.', 'All scores are checkpoint-selection validation, not independent generalization results.']
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--zip',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert hashlib.sha256(args.zip.read_bytes()).hexdigest()==ZIP_SHA
    with TemporaryDirectory() as tmp:
        with ZipFile(args.zip) as z:
            for name in z.namelist():
                assert not Path(name).is_absolute() and '..' not in Path(name).parts
            z.extractall(tmp)
        result=audit(Path(tmp))
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('AUDIT VERIFIED: both seeds; 259 rows each; all saved validation histories reproduced')


if __name__=='__main__': main()
