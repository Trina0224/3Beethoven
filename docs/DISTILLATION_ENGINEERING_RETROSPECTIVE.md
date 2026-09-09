# What It Took to Make a 3B Statistics Student Better

*A response-distillation engineering retrospective: curriculum failures, forgetting, evaluation design, and a bounded success.*

We started with a straightforward hypothesis: if a larger model can solve a well-defined class of problems, a smaller model should be able to learn that capability from its answers. Several days of experiments later, the final Llama 3.2 3B Instruct student scored **170/180 (94.4%)**, compared with **45/180 (25.0%)** for the unmodified model on a reserved statistics-expression evaluation.

The interesting part was the path between those numbers. We saw high scores that did not support broad claims, improved training loss accompanied by lost skills, curricula with many examples but little structural diversity, and a successful-looking model whose adoption we later withdrew. Progress came from improving the supervision and the experiment, not simply running more optimization steps.

## 1. Four examples before the theory

The task is to produce a fully substituted executable expression, leaving arithmetic to a calculator. These four examples were selected from the final evaluation because the original 3B failed and the student succeeded. They illustrate the change; the complete 180-question evaluation establishes its breadth.

**Provenance matters:** the base and student answers below are the recorded final-evaluation outputs. We subsequently asked Llama 3.3 70B Instruct to answer the same unchanged prompts, without supplying a reference answer or student response. Those teacher responses are a **post-training demonstration**, not the training targets for these questions and not an unbiased teacher benchmark. All four teacher expressions are mathematically correct, although none uses the requested `Expression:` prefix. No output has been repaired for presentation.

All questions included this instruction:

> Return one line: Expression: <a fully substituted numerical expression>. Keep arithmetic and unit conversion operations unevaluated. Use integers, fractions, +, -, *, /, ** and comb(n,r). No final answer or explanation.

### Second moment is not squared mean

**Question:** X has a Poisson distribution with mean (177/17). Find E[X**2].

Evaluation ID: `diverse_final_blind_poisson_second_000`

**Original 3B**

```text
Expression: (177/17)**2
```

**70B teacher — post-training response**

```text
(177/17) + (177/17)**2
```

**Final step180 student**

```text
Expression: ((177/17)**2)+(177/17)
```

For a Poisson variable,

$$
E[X^2]=\operatorname{Var}(X)+E[X]^2=\lambda+\lambda^2.
$$

The base omits the variance term. Both teacher and student include it. This is a mathematical improvement, not merely cleaner formatting.

### Knowing when not to convert units

**Question:** Find Var(X). Given a homogeneous Poisson process with rate (23) arrivals per minute, X counts arrivals during (16) minutes.

Evaluation ID: `diverse_final_blind_process_variance_001`

**Original 3B**

```text
Var(X) = (23 * 16) / (60 ** 2)
```

**70B teacher — post-training response**

```text
(23 * 16)
```

**Final step180 student**

```text
Expression: (23)*(16)
```

The rate and duration are already expressed in compatible units:

$$
\operatorname{Var}(X)=\lambda t.
$$

The base introduces an unnecessary division by $60^2$. The student binds rate and duration correctly. Teaching unit conversion must include examples where no conversion is needed.

### Distinguishing neighboring event definitions

**Question:** A and B are independent events. P(B)=(116/131); P(A)=(81/131). Find the probability that both occur or neither occurs.

Evaluation ID: `diverse_final_blind_same_000`

**Original 3B**

```text
P(A and B) + P(neither A nor B) = P(A) + P(B) - P(A and B) = (81/131) + (116/131) - (81/131)*(116/131)
```

**70B teacher — post-training response**

```text
(81/131)*(116/131) + (1 - 81/131)*(1 - 116/131)
```

**Final step180 student**

```text
Expression: (81/131)*(116/131)+(1-(81/131))*(1-(116/131))
```

For independent events with probabilities $p$ and $q$,

$$
P(\text{both or neither})=pq+(1-p)(1-q).
$$

The base initially names the correct events but substitutes the union formula $p+q-pq$, which answers a different question. The student correctly distinguishes “same outcome” from “at least one.” Fluent terminology did not guarantee correct formulation.

### Ignoring irrelevant quantities in a variance question

**Question:** E[X]=(-25/11); Var(X)=(58). Define Y=(-23/13)*X+(-14/23). Find Var(Y).

Evaluation ID: `diverse_final_blind_moment_variance_000`

**Original 3B**

```text
Var(Y)=((-23/13)**2)*Var(X)+((-23/13)**2)*(-25/11)+((-14/23)**2)*Var(X)+((-14/23)**2)*(-25/11)+2*((-23/13)*(-14/23))*(-25/11)
```

**70B teacher — post-training response**

```text
Var(Y)=((-23/13)**2)*(58)
```

**Final step180 student**

```text
Expression: ((-23/13)**2)*(58)
```

For an affine transformation,

$$
\operatorname{Var}(aX+b)=a^2\operatorname{Var}(X).
$$

The base mixes the mean, shift, and cross terms into the answer and leaves `Var(X)` unsubstituted. The student retains only the relevant inputs. Correct binding includes knowing which supplied numbers should not appear in the expression.

## 2. A deliberately bounded interface

We did not ask the student to become a general mathematical reasoner. We asked it to identify the requested quantity, choose the relationship, bind numerical inputs, handle units, and emit an executable expression:

$$
\text{Question}\rightarrow\text{Formulation}\rightarrow\text{Exact calculation}.
$$

A bounded calculator performs the final arithmetic. This separates conceptual mistakes from arithmetic mistakes and avoids spending the same training budget on statistics, long-form explanation, and multi-digit computation.

The final scope covers 18 categories: independent events; affine means, variances, and second moments; Poisson counts and processes; uniform means and conditional means; binomial probabilities; and confidence-interval upper endpoints after sample-size changes. General prose quality and arbitrary unfamiliar paraphrases are outside the success claim.

## 3. What “distillation” meant in this experiment

The main teacher was Llama 3.3 70B Instruct; the student was Llama 3.2 3B Instruct. We used response distillation: discrete target responses provide supervised next-token training signals.

For examples $(x_i,y_i)$, the usual objective is

$$
\mathcal L_{\mathrm{SFT}}=-\sum_i\sum_t\log p_\theta(y_{i,t}\mid x_i,y_{i,<t}).
$$

We did **not** use teacher logits or a temperature-scaled KL objective such as

$$
\mathcal L_{\mathrm{KD}}=T^2D_{\mathrm{KL}}\left(p_T^{(T)}\|p_S^{(T)}\right).
$$

We also did not observe the teacher's internal reasoning state. The outcome measures output capability, not replication of an internal reasoning process.

As the project evolved, we corrected some targets using validated canonical expressions. The final method is therefore best described as **response distillation combined with verified and corrected synthetic supervision and historical replay**. We cannot attribute every improvement specifically to knowledge transferred from the 70B teacher.

The student used LoRA:

$$
W'=W+sBA,
$$

where the low-rank matrices parameterize the update. This kept training and artifact storage manageable. The final adapter is about 97 MB; users load it on the original base rather than downloading a separately redistributed full model.

Parameter efficiency does not protect against bad supervision. A model can learn a flawed curriculum efficiently.

## 4. High scores were not always convincing evidence

An intermediate model reached 96/96 on a reasoning-chain suite. A later fixed-weight diagnostic reported 24/24 on the original questions and 24/24 on magnitude changes, but 15/24 on paraphrases. This did not prove that the model only memorized templates, but it showed that the original score supported a narrower claim than we had assumed.

Another narrow-template experiment improved from v15's 107/144 to 144/144. We later withdrew that model's adoption and success interpretation because the curriculum and evaluation did not adequately cover signs, boundaries, units, and structural combinations. The score remained a historical observation; the broader claim did not survive review.

There were two competing risks. Keeping everything on a familiar template could hide brittleness. Making evaluation language arbitrarily unfamiliar could turn the task into a language-generalization experiment we never intended to run. We ultimately retained direct wording and increased **mathematical structural diversity**.

## 5. Many examples can contain little new information

Some early curricula repeatedly reused a question skeleton and changed only the numbers. That can exercise numerical binding, but it is not equivalent to covering more conceptual decisions.

Hundreds of examples of “rate per minute, duration in minutes, find count variance” can mostly reinforce

$$
\text{Familiar wording}\rightarrow\lambda t.
$$

They do not necessarily teach what changes when duration is in seconds, the target becomes a second moment, an affine transformation is introduced, or an irrelevant mean is supplied.

We moved from accumulating examples to specifying a coverage matrix:

| Dimension | Variations that matter |
|---|---|
| Numbers | Signs, zero, fractions, different probability denominators |
| Units | Seconds/minutes; conversion required versus already compatible |
| Affine transformation | Positive/negative scale; zero/nonzero shift |
| Uniform distribution | Nonzero support lower bound; conditional cutoff; total time |
| Binomial | r=0, 1, interior, n−1, n |
| Confidence interval | Negative/cross-zero endpoints; integer/rational sample-size multipliers |
| Target and ordering | Same givens, different requested quantities; reordered givens |

We also used same-story contrasts. Given $E[X]=\mu$, $\operatorname{Var}(X)=v$, and $Y=aX+b$, the curriculum must distinguish

$$
E[Y]=a\mu+b,
$$

$$
\operatorname{Var}(Y)=a^2v,
$$

$$
E[Y^2]=a^2v+(a\mu+b)^2.
$$

“Contrast” here describes curriculum organization, not a contrastive-loss implementation. The training objective remained SFT and every training target was a correct answer.

### A concrete integrity bug

A binomial story was supposed to include five questions for the same $n,p$, covering $r=0,1,\text{interior},n-1,n$. One generated group stopped after three rows. Aggregate counts and the manifest could still agree with the truncated dataset.

The fix was to check each contrast group's membership and size, rather than trusting total counts and hashes. A hash proves an artifact has not changed; it cannot prove the artifact was correct when created.

## 6. Learning new skills while losing old ones

One intermediate comparison improved new reasoning-chain performance from 72/96 to 96/96, while historical skills changed from 59/96 to 56/96. The small aggregate drop concealed a much larger local collapse: `exactly_one` fell from 10/12 to 1/12.

For independent events,

$$
P(\text{exactly one})=p(1-q)+(1-p)q=p+q-2pq,
$$

whereas

$$
P(\text{at least one})=p+q-pq.
$$

These neighboring concepts differ by one intersection term. Training that reinforces one without preserving the distinction can interfere with the other.

We also observed trajectories in which additional training reduced loss while damaging previously acquired capability:

$$
\text{Lower training loss}\not\Rightarrow\text{broader retained competence}.
$$

This is consistent with interference and catastrophic forgetting, but we did not isolate every mechanism experimentally.

### Retention testing is not replay

A legacy-retention artifact containing only hashes and counts can support evaluation. It does not demonstrate that historical examples were actually used during optimization.

- **Retention evaluation** measures whether old skills survived.
- **Replay/rehearsal** supplies old-skill supervision during training.

The final run materialized 720 historical training rows and interleaved them with 720 new rows. Conceptually, this balances objectives of the form

$$
\mathcal L=\alpha\mathcal L_{\mathrm{new}}+(1-\alpha)\mathcal L_{\mathrm{replay}}.
$$

A 1:1 example ratio is not necessarily an exact 1:1 token-level loss weight because sequence lengths and normalization matter. We did not run an ablation that isolates the replay contribution from the curriculum changes.

## 7. Was the seed simply unlucky?

We investigated training-seed replication for intermediate versions and separately ran fixed-weight input perturbation tests. These answer different questions:

| Diagnostic | What changes | Question answered |
|---|---|---|
| Training-seed replication | Stochastic training trajectory | Does the outcome depend strongly on a particular run? |
| Fixed-weight perturbation | Numbers, ordering, or wording | Does the learned behavior survive input variation? |

Greedy inference removes sampling from decoding, not randomness from training. Depending on the runner, a seed can affect ordering, dropout, or initialization. Replication made it harder to dismiss every failure as bad luck, while perturbations exposed dependence on the input presentation.

These were diagnostic experiments, not a reason to keep searching for the highest-scoring seed. They shifted attention toward coverage, interference, replay, and stopping time. They did not prove curriculum design was the only cause, and intermediate-version replication does **not** establish multi-seed robustness for the final step180 artifact.

## 8. Neither the teacher nor the grader is ground truth

We needed separate checks for prompt semantics, reference formulas, safe execution, structural correctness, and degenerate numerical coincidences.

For example, two inequivalent formulas may produce the same value when a coefficient is zero. Numerical equality at one input is weaker evidence than the correct functional relationship. We therefore used exact arithmetic together with semantic and structural checks, legal-equivalence cases, and near-wrong formula mutations.

A grader should accept both

$$
p(1-q)+(1-p)q
\quad\text{and}\quad
p+q-2pq,
$$

but reject $p+q-pq$ for an exactly-one question. Boundary simplifications also require care: a binomial answer at $r=n$ may correctly omit factors equal to one.

Teacher raw answers and corrected targets were kept distinct. A canonical correction is valid supervision, but it must not be recorded as a correct raw teacher response.

### Zero scores required investigation

Repeated zero scores were a warning to inspect the entire path:

$$
\text{Question/data}\rightarrow\text{Model output}\rightarrow\text{Parser}\rightarrow\text{Equivalence check}\rightarrow\text{Score}.
$$

A zero alone cannot identify whether the model, curriculum, parser, or scorer is at fault. We separated the mathematical primary score from strict-format diagnostics and unresolved review cases.

`primary_correct` measures mathematical/executable success under the grader contract. `strict_one_line_expression` measures interface compliance. `review_required` indicates that the automatic procedure has not established a conclusion. Pending answers should not silently become either proven errors or discretionary successes.

The four teacher examples at the start make the distinction tangible: the teacher's mathematics is correct, but its requested prefix is missing. The student is mathematically correct and follows the requested interface.

## 9. The final training run and the gate it did not pass

| Setting | Final run |
|---|---|
| Starting adapter | v15 |
| New examples | 720 |
| Historical training replay | 720 |
| Ordering | Alternating new/replay |
| Epochs | 1 |
| Optimizer updates | 180 |
| Batch size / accumulation | 1 / 8 |
| Learning rate | 1e-5 |
| Seed | 2027 |
| Checkpoints | 60, 120, 180 |

Development scores progressed from 139/180 to 167/180 to 172/180. The final legacy-development score was 72/72. One owner-authorized provisional credit would make the last new-development score 173/180, but the automatic record remains distinct.

The original v15-based frozen selection protocol required zero paired losses. Step180 still lost two answers v15 had previously answered correctly, so its historical selection result remains `no_checkpoint_passed`.

There were also execution deviations: a baseline completed after training, checkpoint inspection outside the prescribed sequence, and a post-output provisional credit. A later favorable evaluation cannot erase those facts or retroactively make the earlier procedure compliant.

## 10. The final baseline answered a different question

Comparing against v15 answers: “What did this continuation add, and what did it retain?” Comparing against the unmodified 3B answers: “Is the final artifact better than the model we started the project with?”

The owner explicitly requested the latter comparison. We evaluated the original Llama 3.2 3B Instruct and the same base with the already identified step180 adapter on the reserved final set. Although training continued from v15, inference loads the final adapter directly on the base; it does not stack v15 and step180.

Both arms used the same base tokenizer, question prompts, greedy decoding, 160-token generation limit, grader, and 4-bit NF4 configuration. This was a later authorized evaluation, not a retrospective rewrite of the earlier v15 protocol.

## 11. Final results and remaining failures

| Metric | Original 3B | Step180 |
|---|---:|---:|
| Automatically correct | 45/180 (25.0%) | **170/180 (94.4%)** |
| Strict one-line format | 26/180 | **180/180** |
| Pending | 66 | **0** |

| Paired outcome | Count |
|---|---:|
| Wrong → right | 127 |
| Right → wrong | 2 |
| Both correct | 43 |
| Both incorrect | 8 |

The net gain is 125 questions, or 69.4 percentage points. Under automatic scoring, 17 categories improve and one ties. Even crediting all 66 baseline pending answers yields only 111/180, still below 170/180. That is a sensitivity bound, not a completed review of the baseline pending cases.

All ten student failures were inspected and are real formulation errors: three omitted rate/duration terms, three moment/variance mistakes, one interval-width sign error, one missing event complement, one ignored conditional cutoff, and one extra binomial failure-probability factor. Two of those are genuine regressions relative to the base.

This supports a strong **bounded-task improvement**, not perfect reliability, zero forgetting, or universal reasoning capability. Final-blind results were not used to train or select this artifact. Now that the set has been inspected, future model selection needs a new reserved evaluation.

## 12. Reproducibility had practical failure modes too

Kaggle GPU quota exhaustion moved the final evaluation to Colab. Token environment handling, a tokenizer configuration incompatible with the installed Transformers version, and an unrestricted Hugging Face Hub upgrade all caused failures unrelated to model capability.

We fixed the tokenizer source to the pinned original base, constrained incompatible package versions, and used fresh Python subprocesses to avoid mixing notebook-cached modules with newly installed packages. These fixes made the run executable; they were not new training interventions.

We identify the artifact by base revision and adapter SHA-256, not “latest” or a Kaggle notebook version. Code, data, and reports live on GitHub; adapter weights live on Hugging Face.

The four post-training teacher requests returned in 0.24–1.43 seconds and produced only 7–26 completion tokens. API records identify Llama 3.3 70B Instruct, with DeepInfra serving three calls and Groq one. They are therefore not a controlled latency comparison with the Colab student, nor can API metadata independently establish the provider's internal weights.

## 13. What changed in our engineering approach

1. **Treat supervision as an engineered artifact.** Teacher size does not replace correctness checks or provenance.
2. **Specify diversity as coverage.** New numbers are useful, but do not substitute for new decisions about conditions, formulas, units, and boundaries.
3. **Measure and train retention separately.** A retention suite detects forgetting; replay supplies a mechanism intended to reduce it.
4. **Inspect local failures.** Aggregate scores can conceal the collapse of an entire skill.
5. **Separate uncertainty from failure.** Parser limitations and unresolved equivalence cases should remain visible rather than being hidden in a single accuracy number.
6. **Do not seed-shop for a success story.** Replication and perturbations diagnose different risks; neither is a substitute for curriculum repair.
7. **Keep the success claim narrower than the evidence.** We established a large improvement for one artifact on one bounded reserved evaluation, not isolated transfer causality or training-seed robustness.
8. **Stop when the agreed experiment has answered its question.** The remaining errors define possible future work, not a reason to tune indefinitely against an exposed test.

The outcome was more than a smaller model with a better score. We learned how to turn a plausible distillation idea into an inspectable experiment: define the capability, validate the supervision, expose regressions, preserve failed results, and state exactly what the final evidence supports.

## Artifacts and source records

- [Project](../README.md)
- [Final evaluation report](STATS_DIVERSE_FINAL_BLIND_RESULTS.md)
- [Training and development record](STATS_DIVERSE_RUN_RESULTS.md)
- [Curriculum and method overview](STATS_METHOD_OVERVIEW.md)
- [Intermediate seed replication](STATS_SEED_REPLICATION_REPORT.md)
- [Fixed-weight perturbation analysis](STATS_V19_ANALYSIS_REPORT.md)
- [V58 withdrawal](STATS_V58_POSTMORTEM.md)
- [Full base/student evaluation bundle](../final_blind_eval_results.zip)
- [Four post-training teacher examples and API receipts](STATS_TEACHER_FOUR_EXAMPLES.json)
- [Final adapter](https://huggingface.co/kozakurayuki/3Beethoven-step180)

Base revision: `0cb88a4f764b7a12671c53f0838cd831a0843b95`. Adapter SHA-256: `ff89a22f097e4db457dab287042ae36520ec6b9b36f26e5ed11fe42337c04f23`.
