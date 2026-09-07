"""Build the sealed 288-question holdout and fixed 24-question teacher subset.

This module is evaluation-only.  Training and teacher-corpus builders never
import it.  It blocks prior/candidate expressions and parameter identities.
"""
import json
import random
from pathlib import Path

from exact_calculator import calculate
from stats_consolidation_grader import score
from stats_consolidation_pilot import DOCS, build as build_candidate, digest, question
from stats_curriculum_v0_13 import KINDS, make as old_make
from stats_curriculum_v0_18 import TRACKS, make as chain_make
from stats_consolidation_semantics import task_key, validate_question

SEED = 210288


def prior_material():
    expressions, tasks, identities, chain_keys = set(), set(), set(), set()
    for path in sorted(DOCS.glob("STATS_V0_*_FROZEN_QUESTIONS.json")):
        data = json.loads(path.read_text())
        for rows in data.values():
            if not isinstance(rows, list):
                continue
            for q in rows:
                if isinstance(q, dict):
                    if q.get("expression"):
                        expressions.add(q["expression"])
                    try:
                        tasks.add(task_key(q))
                    except (ValueError, KeyError):
                        pass  # Older, unrelated formats still have expression/ID blocking.
                    if q.get("identity"):
                        identities.add(tuple(q["identity"]))
                    if q.get("track") and q.get("parameters"):
                        chain_keys.add((q["track"], tuple(q["parameters"])))
    for story in build_candidate()[0]:
        expressions.update(q["expression"] for q in story["questions"])
        tasks.update(task_key(q) for q in story["questions"])
    return expressions, tasks, identities, chain_keys


def sample_old_parameters(kind, rng):
    # The first slot means a rate for some families, a percentage for others.
    # Never expand a percentage past 100 to manufacture numeric novelty.
    first = rng.randrange(1, 100) if kind in ("binomial", "exactly_one", "at_least_one") else rng.randrange(101, 230)
    return [first, rng.randrange(1201, 3600), rng.randrange(2, 12), rng.randrange(2, 43)]


def reword_old(q):
    b, c = q["bindings"], q["category"]
    if c == "poisson_time":
        return f"A call center is modeled by a homogeneous Poisson process with rate {b['rate_per_minute']} per minute. Over {b['duration_minutes'].split('/')[0]} seconds, represent the count variance."
    if c == "poisson_scaled":
        return f"A latent count X is Poisson with mean {b['mean']}. The reported score is {b['scale']}*X+{b['offset']}. Give Var(score) as one numerical expression."
    if c == "moment":
        return f"A calibrated reading is {b['scale']}*X+{b['offset']}, where E[X]={b['mean']} and Var(X)={b['variance']}. Represent the reading's second moment."
    if c == "uniform_time":
        return f"A shuttle's total wait T is uniform between 0 and {b['upper_minutes']} minutes. Conditional on T exceeding {b['cutoff_minutes'].split('/')[0]} seconds, represent E[T] in minutes."
    if c == "binomial":
        pct = b['reject_probability'].split('/')[0]
        return f"A device performs {b['n']} independent checks, each positive with probability {pct} percent. Represent the probability of exactly {b['r']} positives."
    if c in ("exactly_one", "at_least_one"):
        ma, mb = b['miss_a'].split('/')[0], b['miss_b'].split('/')[0]
        target = "exactly one scanner finds the flaw" if c == "exactly_one" else "one or both scanners find the flaw"
        return f"Independent scanners A and B miss a flaw with probabilities {ma}% and {mb}%. Give the probability that {target}."
    return f"An interval has endpoints {b['lower']} and {b['upper']}. Multiplying sample size by {int(b['width_divisor'])**2} preserves its center and divides its half-width by {b['width_divisor']}. Represent the revised upper endpoint."


def reword_chain(q):
    return ("Evaluation-only scenario. Express the requested quantity from the supplied assumptions: "
            + q["question"].replace("Find", "represent").replace("A homogeneous Poisson process", "Calls follow a homogeneous Poisson process"))


def build():
    rng = random.Random(SEED)
    blocked_expr, blocked_tasks, blocked_identity, blocked_chain = prior_material()
    old, chain, event = [], [], []

    detection_params = []
    for kind in KINDS:
        for i in range(12):
            if kind == "at_least_one":
                p = detection_params[i]
                q = old_make(kind, p, "holdout", i)
            else:
                for _ in range(100000):
                    p = sample_old_parameters(kind, rng)
                    q = old_make(kind, p, "holdout", i)
                    validate_question(q)
                    identity = tuple(q["identity"])
                    probes = [q]
                    if kind == "exactly_one":
                        probes.append(old_make("at_least_one", p, "holdout", i))
                    if (identity not in blocked_identity and
                            not any(x["expression"] in blocked_expr or task_key(x) in blocked_tasks for x in probes)):
                        break
                else:
                    raise RuntimeError("old holdout identity space exhausted")
                if kind == "exactly_one":
                    detection_params.append(p)
            blocked_identity.add(tuple(q["identity"])); blocked_expr.add(q["expression"]); blocked_tasks.add(task_key(q))
            q["id"] = f"consolidation_holdout_old_{kind}_{i:03d}"
            q["question"] = reword_old(q)
            q["provenance"] = "sealed_programmatic_holdout; never_teacher_training"
            old.append(q)

    for track in TRACKS:
        for i in range(8):
            for _ in range(100000):
                p = [rng.randrange(101, 230), rng.randrange(16, 40), rng.randrange(2, 12),
                     rng.randrange(3, 45), rng.randrange(341, 900), rng.randrange(111, 260), rng.randrange(24, 70)]
                probe = [chain_make(track, depth, p, "holdout", i) for depth in (1, 2, 3)]
                if ((track, tuple(p)) not in blocked_chain and
                        not any(q["expression"] in blocked_expr or task_key(q) in blocked_tasks for q in probe)):
                    break
            else:
                raise RuntimeError("chain holdout parameter space exhausted")
            blocked_chain.add((track, tuple(p)))
            for q in probe:
                blocked_expr.add(q["expression"]); blocked_tasks.add(task_key(q))
                q["id"] = q["id"].replace("v18_holdout_", "consolidation_holdout_chain_")
                q["story_id"] = q["story_id"].replace("v18_holdout_", "consolidation_holdout_chain_")
                q["question"] = reword_chain(q)
                q["provenance"] = "sealed_programmatic_holdout; never_teacher_training"
                chain.append(q)

    for i in range(24):
        den = (101, 103, 107)[i % 3]
        while True:
            pa, pb = rng.randrange(7, den - 7), rng.randrange(7, den - 7)
            if pa != pb:
                expressions = {
                    "exactly_one": f"({pa}/{den})*(1-{pb}/{den})+(1-{pa}/{den})*({pb}/{den})",
                    "both": f"({pa}/{den})*({pb}/{den})",
                    "neither": f"(1-{pa}/{den})*(1-{pb}/{den})",
                    "same": f"({pa}/{den})*({pb}/{den})+(1-{pa}/{den})*(1-{pb}/{den})",
                }
                event_specs = {category: {"id": "probe", "category": category,
                    "semantics": dict(kind="events", target=category, p=f"{pa}/{den}", p_b=f"{pb}/{den}")}
                    for category in expressions}
                if (not any(value in blocked_expr for value in expressions.values()) and
                        not any(task_key(q) in blocked_tasks for q in event_specs.values())):
                    break
        story_id = f"consolidation_holdout_event_{i:03d}"
        asks = {"exactly_one": "exactly one signal is on", "both": "both signals are on",
                "neither": "both signals are off", "same": "the signals match"}
        for j, category in enumerate(("exactly_one", "both", "neither", "same")):
            q = question(story_id, category, j,
                         f"Two independent binary signals turn on with probabilities {pa}/{den} and {pb}/{den}. Represent the probability that {asks[category]}.",
                         expressions[category], {}, ["enumerate independent outcomes", category], "a different event")
            q["id"] = f"{story_id}_{category}"
            q["semantics"] = event_specs[category]["semantics"]
            q["provenance"] = "sealed_programmatic_holdout; never_teacher_training"
            blocked_expr.add(q["expression"]); blocked_tasks.add(task_key(q)); event.append(q)

    suites = {"old": old, "chain": chain, "event": event}
    assert {key: len(rows) for key, rows in suites.items()} == {"old": 96, "chain": 96, "event": 96}
    for rows in suites.values():
        for q in rows:
            validate_question(q)
            assert calculate(q["expression"]) == q["answer"]
            assert score("Expression: " + q["expression"], q)["primary_correct"]
    benchmark = ([next(q for q in old if q["category"] == kind) for kind in KINDS]
                 + [q for track in TRACKS for q in [x for x in chain if x["track"] == track][:2]]
                 + [q for category in ("exactly_one", "both", "neither", "same")
                    for q in [x for x in event if x["category"] == category][:2]])
    assert len(benchmark) == 24 and len({q["id"] for q in benchmark}) == 24
    metadata = {"status": "sealed_after_domain_repair_before_model_evaluation", "seed": SEED,
                "data_revision": "consolidation-holdout-v2-domain-repair",
                "supersedes_invalid_holdout_sha256": "c5515b35ff5ee0a5abf7c283c8cab55e08ee2e0e45a0105766865d744ac343df",
                "novelty_rule": "effective subtask identity, not scalar-answer uniqueness",
                "counts": {key: len(value) for key, value in suites.items()},
                "candidate_corpus_sha256": digest(build_candidate()[0]),
                "teacher_benchmark_rule": "old 1/category; chain 2/category; event 2/category",
                "training_use_forbidden": True}
    return {"metadata": metadata, "suites": suites}, benchmark


def main():
    holdout, benchmark = build()
    (DOCS / "STATS_CONSOLIDATION_HOLDOUT.json").write_text(json.dumps(holdout, indent=2) + "\n")
    (DOCS / "STATS_CONSOLIDATION_TEACHER_BENCHMARK.json").write_text(json.dumps(benchmark, indent=2) + "\n")
    print(json.dumps({"holdout_sha256": digest(holdout), "teacher_benchmark_sha256": digest(benchmark),
                      "counts": holdout["metadata"]["counts"]}))


if __name__ == "__main__":
    main()
