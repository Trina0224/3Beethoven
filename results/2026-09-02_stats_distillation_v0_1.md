# Statistics Response-Distillation Run v0.1

Date: 2026-09-02 PT

## Setup

- Teacher: Meta Llama 3.3 70B Instruct via OpenRouter
- Student: Meta Llama 3.2 3B Instruct
- Method: response / synthetic-data distillation
- Training: 4-bit QLoRA on Kaggle T4 GPUs
- Teacher corpus: 120 examples
- Train / validation split: 111 / 9
- LoRA trainable parameters: 24,313,856 / 3,237,063,680 (0.7511%)
- Epochs: 3
- Learning rate: 2e-4
- Gradient accumulation: 8
- Optimizer: paged AdamW 8-bit

## Training behavior

Validation loss improved from 0.573958 after epoch 1 to 0.503331 after epoch 2 and 0.500652 after epoch 3.

Final reported training loss: 0.4557485807509649.

This suggests the adapter learned the synthetic corpus without obvious validation-loss divergence during this short run.

## Frozen targeted benchmark

The frozen evaluation contains 16 questions covering the two capability areas selected because the 70B teacher had previously scored 100% while the vanilla 3B student showed significant headroom:

- distributions / expectation
- inference / hypothesis testing

### Before vs after

| Model / stage | Targeted accuracy |
|---|---:|
| Vanilla Llama 3.2 3B | 56.25% |
| Distilled Llama 3.2 3B | 62.50% |
| Llama 3.3 70B teacher | 100.00% |

Absolute student improvement: **+6.25 percentage points**.

### By category after distillation

| Category | Before | After | Change |
|---|---:|---:|---:|
| Distributions / expectation | 62.5% | 62.5% | 0.0 pts |
| Inference / hypothesis testing | 50.0% | 62.5% | +12.5 pts |

The first small run therefore transferred some inference/testing capability, but did not measurably improve distributions/expectation on this 8-question slice.

## Remaining failure pattern

Post-distillation misses:

- Poisson mean/variance
- linear expectation transformation
- uniform expectation
- Type I error
- Type II error
- confidence-level / interval-width relationship

The student still showed a strong tendency to answer **A** on uncertain multiple-choice items. This pre-existing answer-position bias was not removed by the response-distillation corpus.

## Warnings observed

PEFT emitted gated-repository 401 warnings while attempting to re-fetch the base model config during save/checkpoint operations. These lookups were explicitly reported as being silently ignored, and training completed successfully; the adapter and tokenizer were saved locally.

PyTorch also emitted checkpointing/stream warnings that did not stop training.

## Interpretation

This is a modest but real first transfer result rather than a dramatic success. With only 120 synthetic examples, the student improved from 56.25% to 62.50% on the frozen targeted benchmark, driven entirely by improvement in inference/hypothesis testing.

Important takeaways:

1. Response distillation can move a 3B student with a very small corpus.
2. Teacher quality and curriculum selection matter substantially.
3. A small corpus did not overcome the student's answer-position bias.
4. More data alone should not be assumed to solve the remaining errors; curriculum balance and evaluation design should be improved before scaling.
5. The frozen benchmark must remain excluded from all future training data.

## Next experiment candidates

- Expand only the weak inference/testing concepts with diversified scenarios.
- Add answer-format and multiple-choice-position balancing to reduce A-bias.
- Increase corpus size while preserving concept quotas and near-duplicate filtering.
- Evaluate on an additional fresh held-out benchmark, not only the original 16 questions.
- Compare response distillation with a future logit-based KD experiment when a suitable teacher endpoint exposes logits/soft targets.
