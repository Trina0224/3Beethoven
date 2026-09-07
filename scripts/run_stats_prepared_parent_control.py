"""Isolate numerical preparation using unchanged original v15 weights."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse
import hashlib
import importlib.metadata as md
import json
import sys
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    a = p.parse_args()
    root = a.root
    sys.path.insert(0, str(root / 'repo/scripts'))
    from stats_consolidation_eval import evaluate
    from run_stats_consolidation_compare import suite_metrics
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, prepare_model_for_kbit_training
    from kaggle_secrets import UserSecretsClient
    original = a.input / '3beethoven_first_students'
    contract = read(original / 'original_v15_continued_lora_seed_31415/contract.json')
    parent = a.input / 'first_batch_parent/3beethoven_stats_v0_15/adapter'
    assert hashlib.sha256((parent / 'adapter_model.safetensors').read_bytes()).hexdigest() == contract['parent_adapter_sha256']
    env = {n: md.version(n) for n in ('torch', 'transformers', 'peft', 'bitsandbytes', 'accelerate', 'datasets')}
    assert env == read(root / 'control/environment.json')
    target = root / 'prepared_parent'
    target.mkdir(exist_ok=True)
    token = UserSecretsClient().get_secret('HF_TOKEN')
    modelid = 'meta-llama/Llama-3.2-3B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(modelid, revision=contract['base_revision'], token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(modelid, revision=contract['base_revision'], token=token,
        device_map={'': 0}, torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
    def dtypes(model):
        return {n: str(v.dtype) for n, v in model.named_parameters()
                if n in ('model.embed_tokens.weight', 'model.norm.weight', 'lm_head.weight')}
    before = dtypes(base)
    base = prepare_model_for_kbit_training(base, gradient_checkpointing_kwargs={'use_reentrant': False})
    after = dtypes(base)
    model = PeftModel.from_pretrained(base, parent, is_trainable=True)
    suites = read(root / 'repo/docs/STATS_CONSOLIDATION_SELECTION_VALIDATION.json')['suites']
    result = {'adapter_sha256': contract['parent_adapter_sha256'], 'environment': env,
              'dtypes_before': before, 'dtypes_after': after, 'training_updates': 0,
              'teacher_calls': 0, 'suites': {}, 'pending': []}
    for name, questions in suites.items():
        rows = evaluate(model, tokenizer, questions, target / f'{name}.json')
        save(target / f'{name}_automatic.json', rows)
        old = {r['id']: r for r in read(original / f'v15_selection_rows/{name}.json')}
        for row in rows:
            prior = old[row['id']]
            if row['review_required'] and (row['question_sha256'], row['raw_sha256']) == (prior['question_sha256'], prior['raw_sha256']):
                for key in ('math_correct', 'primary_correct', 'correct', 'review_required', 'reason'):
                    row[key] = prior[key]
                row['semantic_review_reused'] = 'Version53 original v15 identical question/raw hashes'
            if row['review_required']:
                result['pending'].append({'suite': name, **row})
        save(target / f'{name}.json', rows)
        result['suites'][name] = {'before': suite_metrics(list(old.values())), 'after': suite_metrics(rows),
            'same_raw': sum(r['raw_sha256'] == old[r['id']]['raw_sha256'] for r in rows),
            'changes': [{'id': r['id'], 'category': r['category'], 'before': old[r['id']]['raw'],
                         'after': r['raw'], 'before_correct': old[r['id']]['primary_correct'],
                         'after_correct': r['primary_correct']} for r in rows if r['raw_sha256'] != old[r['id']]['raw_sha256']]}
        save(target / 'RESULTS.json', result)
        print('PREPARED_PARENT_SUITE', name, json.dumps(result['suites'][name]), flush=True)
    save(target / 'COMPLETE.json', {'complete': True, 'pending': len(result['pending'])})
    print('PREPARED_PARENT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
