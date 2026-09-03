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
