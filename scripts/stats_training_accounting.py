"""Correct invocation loss accounting for the frozen Transformers 5.0.0 runner.

Trainer resets its loss accumulator on train(), but divides by cumulative
global_step after resume. Preserve that raw value and disclose both denominators.
This is reporting only; it must never affect optimization or checkpoint gates.
"""
import math


def invocation_loss(raw_loss, start_step, end_step, trainer_version='5.0.0'):
    if not 0 <= start_step < end_step:
        raise ValueError('Expected a nonempty forward training invocation')
    if not math.isfinite(raw_loss) or raw_loss < 0:
        raise ValueError('Expected finite nonnegative Trainer loss')
    updates = end_step - start_step
    if trainer_version != '5.0.0':
        return {
            'raw_trainer_training_loss': raw_loss,
            'training_loss': None,
            'training_loss_scope': 'unverified_framework_denominator',
            'invocation_start_step': start_step,
            'invocation_updates': updates,
            'trainer_version': trainer_version,
        }
    return {
        'raw_trainer_training_loss': raw_loss,
        'training_loss': raw_loss * end_step / updates,
        'training_loss_scope': 'current_invocation_mean_per_optimizer_update',
        'invocation_start_step': start_step,
        'invocation_updates': updates,
        'loss_accounting_contract': 'transformers_5.0.0_reset_sum_over_cumulative_step',
    }
