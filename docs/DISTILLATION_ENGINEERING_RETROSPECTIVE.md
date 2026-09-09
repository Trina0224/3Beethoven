# Loss Went Down, Model Got Dumber: Lessons from Distilling Llama 70B into 3B

*A response-distillation engineering retrospective: curriculum failures, forgetting, evaluation design, and a bounded success.*

I started with a hypothesis that sounded almost too obvious to test: if a 70B model can solve a well-defined class of problems, a 3B model should be able to learn that capability from its answers. A few days and about twenty model versions later, the final Llama 3.2 3B Instruct student scored **170/180 (94.4%)** on a reserved statistics-expression evaluation, against **45/180 (25.0%)** for the unmodified 3B.

The score is the least interesting part. Along the way, I saw a 96/96 score lose its shine under perturbation tests, withdrew a later model that scored 144/144, watched a whole skill category collapse from 10/12 to 1/12, and found a curriculum with hundreds of examples that taught almost nothing new. Every real step forward came from fixing the supervision or the experiment, not from running more optimizer steps. This post is about those steps.

## 1. Four examples before the theory

The task: turn a direct statistics question into a fully substituted, executable expression, and leave the arithmetic to a calculator. These four questions come from the final evaluation; the original 3B got them wrong and the student got them right. The full 180-question set establishes breadth; these four show what "better" looks like.

One provenance note. The base and student answers are the recorded final-evaluation outputs. Afterwards I asked Llama 3.3 70B Instruct the same unchanged prompts, with no reference answer and no student output, purely as a demonstration. Those teacher responses were not the training targets for these questions. All four teacher expressions are mathematically correct, and none of them uses the requested `Expression:` prefix. Nothing below has been touched up.

Every question ended with this instruction:

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

The base drops the variance term. Teacher and student both keep it. This is a mathematical fix, not a formatting one.

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

Rate and duration are already in compatible units:

$$
\operatorname{Var}(X)=\lambda t.
$$

The base invents a division by $60^2$. If you teach unit conversion, you also have to teach the cases where no conversion is needed.

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

The base names the right events and then substitutes the union formula $p+q-pq$, which answers a different question. Fluent terminology did not guarantee correct formulation.

<details>
<summary>One more example: ignoring irrelevant quantities in a variance question</summary>

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

The base throws the mean, the shift, and some cross terms into the answer and never substitutes `Var(X)`. The student keeps only what matters. Correct binding includes knowing which supplied numbers should *not* appear.

</details>

## 2. What I actually asked the model to do

I did not ask the 3B to become a general mathematical reasoner. I asked it to identify the requested quantity, choose the relationship, bind the numbers, handle units, and emit one executable expression:

$$
\text{Question}\rightarrow\text{Formulation}\rightarrow\text{Exact calculation}.
$$

A small exact calculator does the arithmetic. That split separates conceptual mistakes from arithmetic mistakes, and it stops me from spending one training budget on statistics, prose, and multi-digit multiplication all at once.

The final scope is 18 categories: independent events (both, neither, exactly one, at least one, same outcome); affine means, variances, and second moments; Poisson counts and processes; uniform means and conditional means; binomial probabilities; and confidence-interval endpoints after sample-size changes. Prose quality and arbitrary paraphrases are outside the claim.

### What "distillation" meant here

Teacher: Llama 3.3 70B Instruct. Student: Llama 3.2 3B Instruct. This was **response distillation**: the teacher's discrete answers became supervised next-token targets, so the loss is the ordinary SFT objective over examples $(x_i,y_i)$:

$$
\mathcal L_{\mathrm{SFT}}=-\sum_i\sum_t\log p_\theta(y_{i,t}\mid x_i,y_{i,<t}).
$$

I did **not** use teacher logits or a temperature-scaled KL objective such as

$$
\mathcal L_{\mathrm{KD}}=T^2D_{\mathrm{KL}}\left(p_T^{(T)}\|p_S^{(T)}\right),
$$

and I never observed the teacher's internal reasoning. The result measures output capability, not replication of a reasoning process.

As the project went on I also corrected some teacher targets against validated canonical expressions, so the honest name for the final method is *response distillation plus verified/corrected synthetic supervision plus historical replay*. I cannot attribute every point of improvement to knowledge transferred from the 70B specifically.

The student was trained with LoRA,

$$
W'=W+sBA,
$$

with rank $r=16$, $\alpha=32$, applied to all seven projection matrices (`q,k,v,o,gate,up,down`). The final adapter is about 97 MB and loads onto the stock base; nobody has to download a redistributed 3B. Parameter efficiency is nice, but it does not protect you from bad supervision. A model can learn a flawed curriculum very efficiently.

## 3. Lesson one: high scores were lying to me

An intermediate model hit 96/96 on a reasoning-chain suite. A later fixed-weight diagnostic on the same weights gave 24/24 on the original questions, 24/24 with the numbers changed, and **15/24 on paraphrases**. That did not prove pure template memorization, but it did prove the 96/96 supported a much narrower claim than I had been making.

Another narrow-template experiment went from v15's 107/144 to a perfect 144/144. I later withdrew that model entirely, because neither the curriculum nor the evaluation adequately covered signs, boundaries, units, or structural combinations. The score stays in the record as a historical observation; the interpretation did not survive review.

There were two ways to get this wrong. Keep everything on a familiar template and brittleness hides. Make the evaluation wording arbitrarily strange and the task silently becomes a language-generalization experiment I never meant to run. I kept direct wording and increased **mathematical structural diversity** instead.

### Many examples, little information

Some early curricula reused one question skeleton and changed only the numbers. That exercises numerical binding, but hundreds of "rate per minute, duration in minutes, find count variance" examples mostly reinforce

$$
\text{Familiar wording}\rightarrow\lambda t.
$$

They provide no explicit coverage of what changes when duration is in seconds, the target becomes a second moment, an affine transform appears, or an irrelevant mean is supplied. So I stopped accumulating examples and started specifying a coverage matrix:

| Dimension | Variations that matter |
|---|---|
| Numbers | Signs, zero, fractions, different probability denominators |
| Units | Seconds/minutes; conversion required versus already compatible |
| Affine transformation | Positive/negative scale; zero/nonzero shift |
| Uniform distribution | Nonzero support lower bound; conditional cutoff; total time |
| Binomial | r=0, 1, interior, n−1, n |
| Confidence interval | Negative/cross-zero endpoints; integer/rational sample-size multipliers |
| Target and ordering | Same givens, different requested quantities; reordered givens |

I also built same-story contrasts. Given $E[X]=\mu$, $\operatorname{Var}(X)=v$, and $Y=aX+b$, the curriculum has to make the model distinguish

$$
E[Y]=a\mu+b,\qquad
\operatorname{Var}(Y)=a^2v,\qquad
E[Y^2]=a^2v+(a\mu+b)^2.
$$

"Contrast" here is curriculum organization, not a contrastive loss. The objective stayed SFT and every target was a correct answer.

### A concrete integrity bug

A binomial story was supposed to contain five questions for the same $n,p$, covering $r=0,1,\text{interior},n-1,n$. One generated group stopped after three rows, and the aggregate counts and manifest still agreed with the truncated dataset. The fix was to check each contrast group's membership and size instead of trusting totals and hashes. A hash proves an artifact has not changed; it says nothing about whether it was right when created.

## 4. Lesson two: loss went down, model got dumber

One intermediate comparison improved new reasoning-chain performance from 72/96 to 96/96, while the historical suite moved from 59/96 to 56/96. Three points, who cares. Except the three points were hiding this: `exactly_one` fell from **10/12 to 1/12**.

For independent events,

$$
P(\text{exactly one})=p(1-q)+(1-p)q=p+q-2pq,
$$

whereas

$$
P(\text{at least one})=p+q-pq.
$$

Neighboring concepts, one intersection term apart. Training that hammers one without preserving the distinction can interfere with the other. I saw several trajectories where more steps meant lower loss and less retained capability:

$$
\text{Lower training loss}\not\Rightarrow\text{broader retained competence}.
$$

This is consistent with interference and catastrophic forgetting. I did not isolate every mechanism experimentally.

### Retention testing is not replay

I had a legacy-retention artifact with hashes and counts, and for a while I treated it as if it protected old skills. It did not. It only measured them.

- **Retention evaluation** tells you whether old skills survived.
- **Replay/rehearsal** actually supplies old-skill supervision during training.

The final run materialized 720 historical training rows and interleaved them 1:1 with 720 new rows, which is conceptually

$$
\mathcal L=\alpha\mathcal L_{\mathrm{new}}+(1-\alpha)\mathcal L_{\mathrm{replay}}.
$$

A 1:1 example ratio is not an exact 1:1 token-level loss weight, because sequence lengths and normalization differ. I also did not run an ablation separating replay from the curriculum changes, so I cannot tell you how much each contributed.

### Was the seed just unlucky?

I ran training-seed replication on intermediate versions and, separately, fixed-weight input perturbations. They answer different questions:

| Diagnostic | What changes | Question answered |
|---|---|---|
| Training-seed replication | Stochastic training trajectory | Does the outcome depend strongly on one run? |
| Fixed-weight perturbation | Numbers, ordering, or wording | Does the learned behavior survive input variation? |

Replication made it hard to blame every failure on bad luck; perturbation exposed how much the behavior depended on presentation. Both pointed back at coverage, interference, replay, and stopping time. Neither was a license to keep rolling seeds until a good story came out.

## 5. Lesson three: neither the teacher nor the grader is ground truth

I needed separate checks for prompt semantics, reference formulas, safe execution, structural correctness, and degenerate numerical coincidences. Two inequivalent formulas can produce the same value when a coefficient is zero, so numerical equality at one input is weak evidence. The grader combines exact arithmetic with semantic and structural checks, legal-equivalence cases, and near-wrong formula mutations.

For an exactly-one question the grader must accept both

$$
p(1-q)+(1-p)q
\quad\text{and}\quad
p+q-2pq,
$$

and reject $p+q-pq$. Boundaries need care too: a binomial answer at $r=n$ may correctly omit factors equal to one.

Teacher raw answers and corrected targets were kept in separate columns. A canonical correction is valid supervision, but it must never be recorded as a correct raw teacher response.

Repeated zero scores were a signal to inspect the whole path, not just the model:

$$
\text{Question/data}\rightarrow\text{Model output}\rightarrow\text{Parser}\rightarrow\text{Equivalence check}\rightarrow\text{Score}.
$$

So the reports carry three separate fields. `primary_correct` is mathematical success under the grader contract. `strict_one_line_expression` is interface compliance. `review_required` means the automatic procedure has not reached a conclusion, and a pending answer must not quietly become either a proven error or a discretionary success. The four teacher examples at the top make the split concrete: the teacher's math is right and its format is wrong; the student gets both.

## 6. The final run and the result

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

Development scores went 139/180 → 167/180 → 172/180, and the legacy-development suite finished at 72/72.

The selection protocol I had frozen before training required *zero* paired losses against v15. Step180 still lost two answers v15 had gotten right, so by that protocol the result is `no_checkpoint_passed`, and it stays that way in the record. There were also execution deviations along the way (a baseline that completed after training, a checkpoint inspected out of sequence, one provisional credit granted after outputs were visible). A good later result does not retroactively make an earlier procedure compliant, so I did not pretend it did.

What I did instead was change the question. Comparing against v15 asks "what did this continuation add and retain?" Comparing against the unmodified 3B asks "is the final artifact better than the model I started with?" The second is the question this project was actually about, so I ran it as a separate, later evaluation on the reserved 180-question set: original Llama 3.2 3B Instruct versus the same base with the already-chosen step180 adapter. Same tokenizer, prompts, greedy decoding, 160-token limit, grader, and 4-bit NF4 loading on both arms. Inference loads step180 directly on the base; v15 is not stacked.

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

Net gain: 125 questions, 69.4 percentage points. Seventeen of eighteen categories improve and one ties. Even if every one of the baseline's 66 pending answers were credited, it would reach 111/180, still well below 170.

I inspected all ten student failures. They are real formulation errors, not parser accidents: three omitted rate/duration terms, three moment/variance mistakes, one interval-width sign error, one missing event complement, one ignored conditional cutoff, and one extra binomial failure-probability factor. Two of them are regressions relative to the base. The final-blind set was never used for training or selection; now that it has been inspected, the next round of model selection needs a fresh reserved set.

## 7. Cost, hardware, and how to try it

Training ran in Kaggle GPU notebooks; the final base-versus-student evaluation ran on a Colab **Tesla T4** (16 GB) after my Kaggle GPU quota ran out. The final run itself is small: 1,440 training rows, one epoch, 180 optimizer updates. I have not reconstructed a complete compute bill for the project, so I would not quote an end-to-end cost from these notebook runs.

The 70B teacher was never loaded locally; it was called through hosted inference APIs. The four demonstration calls in section 1 returned in 0.24–1.43 s with 7–26 completion tokens each, and the API receipts report about **$0.000116 total** for those four calls. That covers only these short demonstration responses, not the full curriculum-generation or training cost. It is not a controlled latency comparison with the T4 student, and API metadata cannot independently prove which weights a provider serves.

Why not MI300 / ROCm? The learning method is not inherently CUDA-specific, but I have not validated this implementation on ROCm. A port would need to check the quantization backend, package compatibility, and inference behavior. If anyone has cycles on an MI-series machine, that would be an interesting next check; the reproduction script is linked at the end.

To run one question on the final student you need the pinned base, the adapter, and the following snippet. Log in to Hugging Face first with an account that has access to the base model (for example, using `huggingface_hub.login()`):

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = "meta-llama/Llama-3.2-3B-Instruct"
BASE_REVISION = "0cb88a4f764b7a12671c53f0838cd831a0843b95"
ADAPTER = "kozakurayuki/3Beethoven-step180"
ADAPTER_REVISION = "682d5555b0e6115135ac7b9b5d718d2abef186de"

# Always use the base tokenizer, not the one in the old Kaggle archive.
tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION)
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.float16,
                           bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained(BASE, revision=BASE_REVISION,
                                            device_map={"": 0},
                                            torch_dtype=torch.float16,
                                            quantization_config=quant)
model = PeftModel.from_pretrained(
    base, ADAPTER, revision=ADAPTER_REVISION, is_trainable=False
).eval()

question = (
    "X has a Poisson distribution with mean (177/17). Find E[X**2].\n"
    "Return one line: Expression: <a fully substituted numerical expression>. "
    "Keep arithmetic and unit conversion operations unevaluated. "
    "Use integers, fractions, +, -, *, /, ** and comb(n,r). No final answer or explanation."
)
chat = tok.apply_chat_template([{"role": "user", "content": question}],
                               tokenize=False, add_generation_prompt=True)
inputs = tok(chat, add_special_tokens=False, return_tensors="pt").to(model.device)
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=160, do_sample=False,
                         pad_token_id=tok.eos_token_id)
print(tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
# Expression: ((177/17)**2)+(177/17)
```

Do not stack the adapter on v15, and do not let the notebook silently upgrade `huggingface_hub`. The tokenizer and package-version problems were frustrating enough without adding an adapter-loading mistake. The three environment failures I hit (token handling, a tokenizer config incompatible with the installed Transformers, an unpinned Hub upgrade) were all unrelated to model capability; the fixes were pinning the tokenizer to the base revision, constraining the two incompatible packages, and running the evaluation in fresh Python subprocesses so cached notebook modules could not mix with newly installed ones. The artifact is identified by base revision and adapter SHA-256, never by "latest" or a notebook version number.

## 8. What changed in how I work

1. **Treat supervision as an engineered artifact.** A bigger teacher does not replace correctness checks and provenance.
2. **Specify diversity as coverage.** New numbers are cheap; new decisions about conditions, formulas, units, and boundaries are what the model needs.
3. **Measure and train retention separately.** A retention suite detects forgetting; replay supplies old-task supervision intended to reduce it.
4. **Inspect local failures.** An aggregate score can hide the collapse of an entire skill.
5. **Separate uncertainty from failure.** Parser limitations and unresolved equivalence cases stay visible instead of disappearing into one accuracy number.
6. **Do not seed-shop for a success story.** Replication and perturbation diagnose different risks; neither substitutes for curriculum repair.
7. **Stop when the agreed experiment has answered its question.** The remaining errors are future work, not a reason to keep tuning against an exposed test.

The outcome was more than a smaller model with a better score. I learned how to turn a plausible distillation idea into an inspectable experiment: define the capability, validate the supervision, expose regressions, keep the failed results, and say exactly what the evidence supports.

## 9. What this post does not claim

I have tried to keep the claim narrower than the evidence, so here is the boundary in one place. This is a large improvement for **one artifact** (step180) on **one bounded, reserved evaluation** of 18 direct-wording statistics categories. It is not perfect reliability (ten failures, two regressions), not zero forgetting, not general mathematical reasoning, and not robustness to unfamiliar paraphrase. The intermediate-version seed replication says nothing about the seed robustness of step180 itself. I cannot separate the contribution of the 70B teacher from the corrected targets and the replay, because I did not run that ablation. And by the protocol I froze before training, the final checkpoint did not pass; the headline number comes from a separate, later comparison against the original base that I chose to run because it answers the question I actually cared about.

## Artifacts and source records

- [Project](../README.md)
- [Final evaluation report](STATS_DIVERSE_FINAL_BLIND_RESULTS.md)
- [Colab reproduction script and handoff](STATS_DIVERSE_FINAL_BLIND_COLAB_HANDOFF.md)
- [Training and development record](STATS_DIVERSE_RUN_RESULTS.md)
- [Curriculum and method overview](STATS_METHOD_OVERVIEW.md)
- [Intermediate seed replication](STATS_SEED_REPLICATION_REPORT.md)
- [Fixed-weight perturbation analysis](STATS_V19_ANALYSIS_REPORT.md)
- [V58 withdrawal](STATS_V58_POSTMORTEM.md)
- [Full base/student evaluation bundle](../final_blind_eval_results.zip)
- [Four post-training teacher examples and API receipts](STATS_TEACHER_FOUR_EXAMPLES.json)
- [Final adapter](https://huggingface.co/kozakurayuki/3Beethoven-step180)

Base revision: `0cb88a4f764b7a12671c53f0838cd831a0843b95`. Adapter SHA-256: `ff89a22f097e4db457dab287042ae36520ec6b9b36f26e5ed11fe42337c04f23`.
