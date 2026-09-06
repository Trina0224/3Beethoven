# v0.7 inference-only diagnostic protocol

This diagnoses baseline Llama 3.2 3B and the leading v0.5 adapter. No training and no teacher API calls. v0.6 is not a comparison model in this diagnostic.

Use the exposed v0.6 questions at indices 0,1,4,5 within each of six topics: 24 questions, selected by index before diagnostic generation. These questions are development material, not a new held-out benchmark.

For each model and question, generate:
- Original multiple-choice prompt under four cyclic rotations (96 total).
- Supplied correct numerical value, then match it to one of four rotated options (96). This intentionally supplies the answer to test mapping/instruction following.
- No options, direct numerical answer (24).
- No options, short calculation followed by numerical answer (24).
- No options, supplied correct rule followed by numerical answer (24).
- Correct substituted numerical expression only (24).

Total: 288 responses/model, 576 responses. Exact prompts and references are saved in tasks.json before generation. Match the earlier base revision, NF4 quantization and saved v0.5 adapter hash. Greedy decoding; max new tokens 16 for letters, 48 for direct numeric tasks, 192 for short calculations. Preserve every raw answer, parsing failure and token-limit hit.

Numeric scoring accepts explicit final Answer lines or a standalone number/fraction. Tolerance is max(1e-6, abs(reference)*1e-4). Do not fish arbitrary numbers from explanatory prose. Report strict results first and inspect unparsed/limited answers manually; distinguish format failures from mathematical errors. Match original MC raw outputs against v0.6's same-model results to check reproduction.

Interpretation:
- Correct value matching with poor unassisted MC weakens the hypothesis of a basic letter-mapping inability.
- Improvement with supplied rules is consistent with difficulty choosing/applying the rule, but prompts and task difficulty also change.
- Failure even on substituted arithmetic points toward arithmetic or output-format difficulties; audit raw text.
- Short calculations can reveal observable mistakes; they do not expose private internal computation.
- These aids do not establish causal localization or improve a trained model. No aggregate should be presented as new generalization performance.
- Audit numeric results by topic, question and condition, not just overall averages. The small, parameterized sample limits scope.
