"""Bind cached baseline scores to the numerical evaluation configuration."""
import hashlib
import json

NUMERICAL_PREPARATION = 'nf4_fp16_compute_kbit_prepared_peft_trainable_v1'


def baseline_manifest(metrics, context):
    return {'version': 1, 'numerical_preparation': NUMERICAL_PREPARATION,
            'context': context,
            'metrics_sha256': hashlib.sha256(json.dumps(metrics, sort_keys=True).encode()).hexdigest()}


def validate_baseline_manifest(metrics, manifest, context):
    if manifest != baseline_manifest(metrics, context):
        raise RuntimeError('Cached v15 baseline lacks the matching numerical/data contract. '
                           'Use a new output root to regenerate it; do not reuse or overwrite historical scores.')
