# 3Beethoven Project Specification

## Active statistics pilot addendum — 2026-09-06 PDT

Latest evaluated run: **v0.18**. New-chain reviewed v15/staged/shuffled scores: 76/96, 96/96, 96/96. Historical eight-family reviewed scores: 64/96, 36/96, 44/96. No promotion. All 35 pending transfer responses reviewed; 26 receive mathematical equivalence credit. Final full Kaggle preservation is not yet confirmed.

Authorized next run: **v0.19**, full eight-family historical teacher replay plus v18 procedural chain supervision from exact v15. Record the hypothesis and retention/selection gates before training; see [v19 protocol](docs/STATS_V0_19_PROTOCOL.md). New teacher calls remain zero; logit KD remains deferred. This repair does not isolate a replay-only causal effect.

Previous completed run: **v0.17**. Exact v15/v16 mixtures and conservative retraining were evaluated using validation-only candidate selection. On the same new 96-question unaided test, v15/v16/25%-v16-mixture/retrained-step8 score 64/57/65/64. The selected student's old MC remains 127/240. Neither approach meets the promotion target; keep v15 as general candidate and preserve complementary v16 capability. v16's second-moment and affine-Poisson-variance scores are 9/12 and 12/12 in this new prompt/test condition. This does not isolate prompting from question differences. Training completed 32 steps on 256 accepted historical teacher responses; no new teacher calls. See `docs/STATS_V0_17_REPORT.md`.

Previous completed run: v0.16, 150 steps with same-story contrasts and fading formula support. On the same new 96-question unaided test, reviewed v15/v16 scores are 34/29; with focused symbolic rules on 24 target-family questions they are 11/22. v16 is not promoted as a general replacement: retain v15 as the general candidate and preserve v16 for rule-assisted formulation. Low validation loss did not ensure broad transfer. See `docs/STATS_V0_16_REPORT.md`; prior v15 results below remain historical facts on their original test.

The original classical-music specification below is retained as project history and application intent. The user-authorized active pilot now targets **correct mathematical formulation**: understand a word problem, identify the event and units, bind the correct numbers, and emit an executable expression. Final arithmetic is performed by a bounded exact calculator. Merely naming a formula or giving the right final number does not meet this objective.

v0.15 training is complete: 200 training examples, 30 validation, 75 steps, selected checkpoint-75. On the same new 64-question test, vanilla/v0.14/v0.15 semantic counts are 8/30/44 (automatic 8/30/42). Poisson time and conditional wait improve, while second moments and affine Poisson variance remain 0/8. Teacher moment targets were verified, but student transfer remains unsolved. The run combines decomposition, more weak-family practice, focused symbolic teacher reminders and replay; it does not isolate any one factor. Preserve all failed teacher attempts and separate initial from supplemented acceptance. Reserve test questions from teacher generation and checkpoint selection. Weights are saved in Kaggle Version 29.

Teacher text is not ground truth, and text explanations are not access to internal reasoning. Preserve rejected attempts, report reference conditioning and format-only corrections, and compare student checkpoints on identical questions and prompts. Logit-based distillation remains deferred. Current evidence and backup locations are recorded in [STATS_CURRENT_STATUS.md](docs/STATS_CURRENT_STATUS.md).

## 1. One-line concept

**Distill a large cloud Llama's classical-music knowledge and explanatory style into a small local Llama specialist that is technically credible and intentionally funny.**

## 2. Research question

Can a roughly 3B-parameter local Llama become a materially better classical-music specialist after response distillation from a much larger cloud Llama teacher, while remaining small enough for practical local inference?

## 3. Why this domain

Classical music is a good first distillation domain because much of the core knowledge is mature, stable, extensively documented, and separable from subjective taste. That makes it easier to build reliable training examples and held-out evaluation than open-ended troubleshooting or expert-system domains.

The project intentionally prioritizes:

1. a memorable/funny demo,
2. a defensible experiment,
3. measurable learning,
4. practical usefulness only after the above.

## 4. Initial model plan

### Teacher

A large cloud-hosted Meta Llama model, preferably around the 70B class if access and cost are reasonable.

The teacher is used to generate synthetic training examples and structured explanations. It is not assumed to be infallible.

### Student

A local Meta Llama model in approximately the 3B class.

The first baseline is the unmodified student. The trained artifact is the same student architecture after response-distillation training/fine-tuning on filtered teacher-generated examples.

### Why not 1B first

A 1B student may be useful for extremely narrow tasks, but the intended project combines factual knowledge, terminology, classification, stylistic comparison, and explanation. A ~3B student offers a better balance between local deployability and domain capacity for the first experiment.

## 5. Distillation method — phase 1

This project begins with **response distillation / synthetic-data distillation**.

Pipeline:

1. Define a bounded classical-music competency map.
2. Generate prompts/tasks from that map.
3. Query the large Llama teacher.
4. Store structured teacher responses.
5. Validate/filter records.
6. Split train/validation/test with leakage controls.
7. Fine-tune the local student on teacher responses.
8. Compare vanilla student, distilled student, and teacher on held-out tasks.

This is deliberately different from logit-based KD; the latter is reserved for a later project phase.

## 6. Competency map — draft v0

The first dataset should cover a bounded set of skills rather than "all classical music."

### A. Period recognition

- Baroque
- Classical
- Romantic
- late Romantic / transition into early modernism where appropriate
- selected early modern / 20th-century context if needed

Tasks:

- identify period from factual/stylistic clues
- order periods chronologically
- reject anachronisms

### B. Composer knowledge

Core representative composers may include Bach, Handel, Vivaldi, Haydn, Mozart, Beethoven, Schubert, Chopin, Schumann, Brahms, Tchaikovsky, Wagner, Verdi, Mahler, Debussy, Ravel, and selected others needed for balanced coverage.

Tasks:

- composer-period mapping
- representative work matching
- factual comparison
- common misconception detection

### C. Musical forms and genres

Examples:

- fugue
- sonata / sonata form
- symphony
- concerto
- rondo
- theme and variations
- opera
- string quartet
- tone poem

Tasks:

- definition
- recognition from description
- comparison between forms
- composer/work examples

### D. Instrumentation and orchestration

Tasks:

- instrument-family knowledge
- ensemble/orchestra terminology
- historically reasonable orchestration distinctions
- differences in broad orchestral practice across periods

Avoid unverifiable aesthetic judgments.

### E. Terminology

Examples:

- tempo and expression terms
- dynamics
- articulation
- texture
- harmony-related vocabulary at an accessible but musically correct level

### F. Style comparison

Tasks should ask for observable or historically grounded differences, not subjective rankings.

Example:

- compare Mozart and Mahler in broad orchestral scale and historical context
- contrast Baroque contrapuntal practice with later Classical textures

### G. Misconception detection

This is intentionally demo-friendly.

Examples:

- "Bach was a Romantic composer."
- "Beethoven wrote The Four Seasons."
- "A fugue is simply any fast orchestral movement."

The model should identify the factual error, correct it, and briefly explain why.

## 7. Humor/personality layer

The distilled model may have a playful persona: a tiny local model with the confidence of an intolerable conservatory know-it-all.

However:

- correctness comes first,
- humor should be separable from factual content,
- evaluation should score facts independent of jokes,
- persona examples must not dominate the dataset,
- the student should be able to answer plainly when requested.

Possible demo persona line:

> "You called Bach Romantic. My three billion parameters would like a word."

This is presentation flavor, not an evaluation criterion.

## 8. Dataset record design — draft

Prefer structured JSONL records such as:

```json
{
  "id": "period_000123",
  "category": "period_recognition",
  "difficulty": "medium",
  "prompt": "A composer active around 1720 writes dense contrapuntal keyboard works and church cantatas. Which broad period best fits?",
  "answer": "Baroque",
  "explanation": "The chronology and emphasis on contrapuntal keyboard and sacred vocal writing are characteristic of the Baroque period.",
  "key_facts": ["c.1720", "counterpoint", "church cantatas"],
  "teacher_model": "<model-id>",
  "generation_version": "v1",
  "validation_status": "pending"
}
```

A separate presentation/persona field can be introduced later if useful.

## 9. Validation strategy

Teacher outputs must be checked before becoming training data.

Possible validation layers:

1. deterministic checks for dates, mappings, and structured fields,
2. comparison against trusted public reference data,
3. second-pass teacher critique where useful,
4. manual spot checks,
5. duplicate/near-duplicate detection,
6. ambiguity filtering.

The project should prefer fewer high-quality records over a huge noisy synthetic dataset in the first experiment.

## 10. Evaluation design

Create a held-out benchmark before or independently from final training generation.

Suggested metrics:

- exact-match accuracy for factual mappings
- classification accuracy
- macro F1 where classes are imbalanced
- misconception correction accuracy
- rubric-based explanation score
- factual hallucination/error rate
- response latency
- peak memory usage
- model/storage size

Minimum comparison:

| Model | Role |
|---|---|
| Vanilla local Llama ~3B | Baseline student |
| Distilled local Llama ~3B | Experimental student |
| Large cloud Llama | Teacher/reference |

## 11. Success criteria — initial

Do not hard-code a required percentage before seeing baseline difficulty. The first experiment is successful if it demonstrates all of the following:

1. statistically/operationally meaningful improvement over the vanilla student on held-out classical-music tasks,
2. clear retained advantage in local inference cost/latency/footprint versus the cloud teacher,
3. no evidence that gains come only from train/test leakage,
4. documented failure cases and limitations,
5. reproducible training and evaluation steps.

## 12. Out of scope for phase 1

- audio understanding or direct music recognition from recordings
- score-image/OCR analysis
- MIDI generation
- music composition quality evaluation
- subjective "best composer" judgments
- RAG as the main mechanism
- AMD/company-internal knowledge
- NDA material
- Linux/kernel debugging
- general-purpose agent routing
- claims that the student is equivalent to a professional musicologist

## 13. Future experiments

### Phase 2: logit-based KD

Compare response-only distillation with classic soft-target KD using teacher logits/logprobs where a suitable model/provider permits it.

Potential variables:

- temperature
- forward/reverse KL variants
- soft-target vs hard-target mixtures
- top-k logit storage

### Phase 3: retrieval augmentation

Optionally explore whether a distilled specialist plus RAG performs better than either approach alone for obscure factual material.

### Phase 4: audio/multimodal extension

Only after the text specialist is stable, consider audio descriptors, symbolic music, MIDI, score analysis, or multimodal models.

## 14. Deliverables

Initial target deliverables:

- reproducible teacher-data generation pipeline
- validated classical-music synthetic dataset
- baseline benchmark
- distilled local student
- comparison report/plots
- concise interactive demo
- documentation explaining what did and did not transfer

## 15. Project tone

The repo may be playful. The experiment must not be sloppy.

**3Beethoven should be funny enough that people want to try it, and rigorous enough that an ML engineer can inspect the methodology without immediately finding an obvious hole.**


## Motivation and expectations for future statistics versions

Before generating data or starting a new version, record the triggering observation, why the change is proposed, the expected direction of change, what would weaken that expectation, the comparison/control, stopping and selection rules, and scope limitations. Commit that record before execution. Keep original expectations immutable; timestamp amendments and state what results were already visible. Append outcomes afterward rather than rewriting the original motivation.

Researcher recollections and later narrative reconstruction must be labeled retrospective. The v15–v18 source audit, research narrative and reusable recording fields are in [STATS_MOTIVATION_EXPECTATIONS.md](docs/STATS_MOTIVATION_EXPECTATIONS.md).
