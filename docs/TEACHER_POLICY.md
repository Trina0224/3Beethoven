# Teacher Policy and Educational Scope

## Why the public experiment uses Llama 70B

This repository is an educational model-distillation experiment. The public version intentionally uses a Meta Llama teacher for synthetic-data generation because the project must respect the usage terms of model providers.

OpenAI and Anthropic public API terms may restrict using their model outputs as training targets for competing general-purpose or generative models. Therefore, this repository does **not** use personal/public ChatGPT or Claude API outputs to train the Llama student unless an applicable license or written authorization explicitly permits it.

The choice of Llama 3.3 70B as the public teacher should **not** be interpreted as a claim that it is the strongest available teacher. In fact, the experiment has already shown meaningful teacher limitations in some domains. That limitation is part of the educational value of the project.

## Intended interpretation

The public experiment demonstrates the **distillation pipeline and methodology**:

1. benchmark the teacher before trusting it;
2. benchmark the student before training;
3. identify domains where the teacher is strong and the student is weak;
4. generate bounded synthetic training data;
5. validate/filter the data;
6. fine-tune the local student;
7. evaluate the student on frozen held-out benchmarks;
8. document gains, failures, teacher errors, and leakage controls.

The public result therefore answers:

> What can be transferred from an accessible, legally usable 70B teacher into a local ~3B student using response distillation?

It does **not** claim to establish the maximum capability achievable with the best frontier teacher.

## Frontier-teacher follow-up

Readers who have legitimate access and permission to use a stronger teacher may repeat the same pipeline with a frontier model and compare the result.

A stronger legally authorized teacher could include, depending on the user's own contracts and permissions:

- an enterprise/internal gateway with explicit rights to use generated outputs for model training;
- another model whose license explicitly permits output-based distillation;
- a privately hosted or open-weight model with compatible terms.

The repository intentionally leaves this as a follow-up experiment rather than bypassing provider restrictions.

A useful comparison would be:

| Experiment | Teacher | Student |
|---|---|---|
| Public baseline | Llama 3.3 70B | Llama 3.2 3B |
| Authorized frontier follow-up | stronger permitted teacher | same Llama 3.2 3B |

Keeping the student, benchmark, curriculum, and training recipe constant would isolate the effect of **teacher quality**.

## Important project lesson

Teacher parameter count alone does not guarantee teacher quality. The project has already observed that a 70B model can be strong in one domain and unreliable in another. For that reason, **teacher competency must be measured before synthetic data generation begins**.

This constraint is intentional: the public repository teaches a reproducible, legally cautious distillation workflow. Repeating the same workflow with a stronger authorized teacher is left to the reader.

## Clarification checked against official terms (2026-09-06 UTC)

The [individual Terms of Use](https://openai.com/policies/row-terms-of-use/), effective January 1, 2026, restrict using Output to develop competing models. They do not state a blanket exemption merely because a project is small, educational, or noncommercial.

The [Services Agreement](https://openai.com/policies/services-agreement/), sections 3.3 and 16 definitions, contains specified exceptions for certain non-distributed classification/organization models and customization of models supplied through OpenAI's services. Those exceptions should not be assumed to cover this generative Llama project. Whether a particular project is competing requires attention to the applicable agreement and use; this document is not a legal determination that every third-party fine-tune is prohibited.

For this repository, retain the explicit Llama-only training-target policy unless applicable permission is established. Do not describe OpenAI assistance with engineering or reporting as OpenAI-generated teacher targets; record actual provenance. No ChatGPT-generated answer explanations were added as student targets in v0.4.

The present experiment also does not show that the 70B teacher is inadequate: on the same original-order 60-item holdout, the teacher scored 52/60 versus the untrained student's 21/60. The training set contained only 48 independent training questions across six narrow templates. Coverage, target weighting, and student optimization remain plausible limitations, not established single causes.
