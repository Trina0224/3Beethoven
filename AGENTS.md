# AGENTS.md

## Project: 3Beethoven

This repository is an AI/ML experiment in **response distillation**: use a large cloud-hosted Llama teacher to generate high-quality classical-music training data, then train a much smaller local Llama student to become a focused classical-music specialist with a humorous personality.

Read this file before modifying the repository.

## Non-negotiable Git rules

1. **Do not add `Co-authored-by:` trailers** to commits.
2. Do not add AI names, model names, assistant names, or bot identities as commit authors, co-authors, sign-offs, or trailers unless the repository owner explicitly asks.
3. Keep commit authorship attributable only to the human repository owner/account performing the commit.
4. Do not rewrite Git history, force-push, rebase shared history, amend unrelated commits, or alter author metadata unless explicitly requested.
5. Use concise normal commit messages. Do not add generated-by-AI boilerplate.
6. Do not create pull requests, branches, releases, tags, or GitHub Actions unless explicitly requested or clearly required by an approved implementation plan.

## Data and confidentiality rules

1. This repository is for **public/non-confidential material only** unless the owner explicitly moves the work into an approved company environment.
2. Never add AMD NDA material, internal documentation, proprietary debug guides, customer information, private source code, private logs, credentials, API keys, tokens, or secrets.
3. Do not use confidential work knowledge to generate training data.
4. Prefer public-domain, openly licensed, or clearly permitted sources for musicology facts and evaluation data.
5. Keep source provenance for externally sourced datasets or reference material.

## Model direction

### Current primary path

- Teacher: large cloud-hosted **Meta Llama** model.
- Student: small local **Meta Llama**, initially targeting roughly the 3B class.
- Method: **response distillation / synthetic-data distillation**.
- The teacher generates structured high-quality responses/examples.
- The student is trained on filtered teacher outputs.
- Evaluation compares the vanilla student, distilled student, and teacher.

### Future path — preserve, but do not implement yet

A later experiment may explore **logit-based knowledge distillation** using teacher logits / soft targets, temperature scaling, softmax distributions, KL divergence, and a distillation loss. Do not mix this into the first experiment unless explicitly requested.

## Product/research intent

The project should feel funny and memorable on the surface, but technically defensible underneath.

The desired character is a **tiny local classical-music snob / specialist**: knowledgeable, slightly overconfident for comedic effect, but factual performance must be evaluated independently from personality.

Humor must never be used to hide weak evaluation.

## Scope discipline

The first version should specialize in classical music rather than general music. Prioritize mature, well-documented musicology topics where answers can be evaluated reliably, such as:

- historical periods and chronology
- composers and representative works
- musical forms
- instrumentation and orchestration concepts
- terminology
- stylistic comparison
- work/composer/period relationships
- detecting common factual misconceptions

Avoid subjective claims such as ranking composers by greatness or declaring one work objectively better than another.

## Evaluation principles

The project must include a held-out evaluation set that is not used to generate training examples.

Prefer measurable tasks such as:

- factual QA accuracy
- era / composer / form classification
- work-composer matching
- misconception detection
- structured rubric scoring
- hallucination/error rate
- latency
- memory use
- model size

Do not claim success based only on cherry-picked chat examples.

When feasible, compare at least:

1. Vanilla local student
2. Distilled local student
3. Cloud teacher

## Teacher-data generation rules

1. Teacher output is not automatically ground truth.
2. Generate structured records that can be validated.
3. Separate facts from humorous persona text when possible.
4. Filter inconsistent, unsupported, ambiguous, or low-confidence records before training.
5. Preserve a clean machine-readable dataset format.
6. Prevent train/test leakage.
7. Track prompts, model identifiers, generation parameters, and dataset versions for reproducibility.

## Engineering style

- Prefer Python for training/evaluation utilities unless another language is clearly better.
- Keep scripts small and composable.
- Use configuration files for model IDs, paths, and experiment parameters rather than hard-coding them.
- Never commit secrets. Use environment variables and provide `.env.example` when needed.
- Make reproducibility a first-class goal.
- Record dependencies explicitly.
- Add tests for deterministic parsing/scoring utilities.

## AI-agent behavior

Before making significant changes:

1. Read `AGENTS.md`.
2. Read `PROJECT_SPEC.md`.
3. Inspect the current repository state.
4. Preserve existing design decisions unless the owner asks to change them.
5. Prefer a minimal, working experiment over an oversized framework.

If a requirement is ambiguous, choose the interpretation that keeps the first experiment smaller, safer, and easier to evaluate.

Do not turn this project into a generic RAG application, generic chatbot, agent framework, or Linux/debugging assistant unless explicitly requested. Those are separate possible projects/iterations.
