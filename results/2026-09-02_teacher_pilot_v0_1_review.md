# Teacher Pilot v0.1 Review

Date: 2026-09-02 PT

Teacher: Llama 3.3 70B Instruct via OpenRouter
Target student: Llama 3.2 3B Instruct
Purpose: validate synthetic-data quality before any distillation training.

## Generation outcome

- Requested: 60 examples
- Generated successfully: 57
- JSON parse failures: 3
- Success rate: 95%

## Category allocation

- harmony_counterpoint: 30 requested
- orchestration: 12 requested
- form_analysis: 10 requested
- history_context: 5 requested
- style_comparison: 3 requested

## Important finding

The pilot data is **not approved for training**.

The 70B teacher produced several responses that are plausible-sounding but musicologically imprecise or incorrect. This is exactly the kind of failure that synthetic-data distillation can amplify if teacher outputs are trusted blindly.

Examples observed in the first five samples:

1. Fugue vs canon explanation overstated that canon entries occur at the same pitch or octave. Canons may imitate at other intervals.
2. The comparison of Renaissance vs Classical suspension treatment was too loose and suggested Classical suspensions may resolve by leaps in a way that obscures the defining prepared-dissonance/resolution behavior.
3. The suspension/appoggiatura answer called both 'dissonant consonances', which is internally contradictory terminology.
4. Multiple generated questions were near-duplicates, especially fugue-vs-canon and suspension topics.
5. Some prompts drifted outside the intended classical-music scope, e.g. comparing jazz and classical modal interchange.

## Decision

Do not train on `teacher_pilot_v0_1.jsonl`.

Revise the generator before scaling up. The next version should add:

- strict JSON schema / structured output handling
- automatic retry on malformed JSON
- semantic or lexical de-duplication
- stronger domain constraints
- teacher self-critique / verification pass
- explicit rejection of ambiguous or controversial claims
- checks against benchmark leakage
- quality labels so rejected rows never enter training

## Why this matters

This is an important part of the experiment: response distillation does not merely copy capability; it can also copy confident errors. Teacher-data quality control is therefore part of the distillation system, not an optional cleanup step.
