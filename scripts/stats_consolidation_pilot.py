"""Build the reviewable consolidation corpus candidate without teacher calls.

The teacher request file intentionally omits references.  Concrete holdout
questions are not built here; only their separately owned blueprint is emitted.
"""
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from fractions import Fraction as F
from pathlib import Path

from exact_calculator import calculate
from formulation_grader import grade
from stats_curriculum_v0_13 import KINDS

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
SEED = 210021
TEACHER_MODEL = "meta-llama/llama-3.3-70b-instruct"
ARCHETYPES = (
    ("poisson_process", 12, 3),
    ("affine_poisson", 12, 3),
    ("general_moment", 12, 3),
    ("uniform_wait", 8, 2),
    ("binomial", 8, 2),
    ("detection_events", 16, 4),
    ("interval", 8, 2),
    ("chain_poisson_variance", 8, 2),
    ("chain_scaled_variance", 6, 2),
    ("chain_second_moment", 6, 2),
)
SURFACES = (
    "direct_definition", "conditions_first", "quantity_first",
    "operational_story", "symbolic_story", "unit_emphasis",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def question(story_id, category, index, text, expression, bindings, rules, confused_with):
    q = {
        "id": f"{story_id}_{index:02d}",
        "story_id": story_id,
        "category": category,
        "question": text,
        "bindings": bindings,
        "expression": expression,
        "answer": calculate(expression),
        "conceptual_rules": rules,
        "confused_with": confused_with,
        "provenance": "independent_programmatic_reference; teacher_has_not_been_called",
    }
    judged = grade("Expression: " + expression, q)
    if judged["math_correct"] is not True or not judged["executable"]:
        raise AssertionError((q["id"], judged))
    return q


def apply_surface(text, style):
    sentences = [part.strip() for part in text.split(". ") if part.strip()]
    if style == "direct_definition":
        return text
    if style == "conditions_first":
        return "Conditions first: " + text
    if style == "quantity_first":
        if len(sentences) > 1:
            return sentences[-1].rstrip(".") + ". Given: " + ". ".join(sentences[:-1]).rstrip(".") + "."
        return "Requested quantity first. Given: " + text
    if style == "operational_story":
        return "During an operations review, " + text[0].lower() + text[1:]
    if style == "symbolic_story":
        return "Write a symbolic numerical setup for this situation. " + text
    if style == "unit_emphasis":
        return text + " Keep every stated unit explicit in the expression."
    raise ValueError(style)


def bundle(archetype, split, i, rng):
    story_id = f"consolidation_{split}_{archetype}_{i:03d}"
    style = SURFACES[i % len(SURFACES)]
    number_form = "integers"
    qs = []
    if archetype == "poisson_process":
        rate, seconds, scale = rng.randrange(12, 70), rng.randrange(71, 719), rng.randrange(2, 8)
        mean = f"{rate}*({seconds}/60)"
        qs.append(question(story_id, "poisson_time", 0,
            f"A service counter receives arrivals at {rate} per minute. During a {seconds}-second observation, what is the variance of the arrival count?",
            mean, {"rate_per_minute": str(rate), "duration_minutes": f"{seconds}/60"},
            ["convert seconds to minutes", "Poisson variance equals its mean"], "rate times seconds without conversion"))
        qs.append(question(story_id, "v18_second_moment", 1,
            f"At the same counter, X is the count during {seconds} seconds. Set up E[X**2].",
            f"({mean})+({mean})**2", {},
            ["Poisson variance equals its mean", "second moment is variance plus squared mean"], "variance alone"))
    elif archetype == "affine_poisson":
        lam, scale, offset = rng.randrange(11, 91), rng.randrange(2, 8), rng.randrange(3, 29)
        qs.extend([
            question(story_id, "poisson_scaled_mean", 0,
                f"X is Poisson with mean {lam}. A display reports Y={scale}*X+{offset}. Set up E[Y].",
                f"{scale}*{lam}+{offset}", {}, ["affine expectation"], "variance scaling"),
            question(story_id, "poisson_scaled", 1,
                f"For that display Y={scale}*X+{offset}, set up Var(Y).",
                f"{scale}**2*{lam}", {"mean": str(lam), "scale": str(scale), "offset": str(offset)},
                ["Poisson variance equals mean", "constant offset has zero variance", "variance scales by the square"], "scale rather than squared scale"),
            question(story_id, "poisson_scaled_moment", 2,
                f"For the same Y={scale}*X+{offset}, set up E[Y**2].",
                f"{scale}**2*{lam}+({scale}*{lam}+{offset})**2", {},
                ["variance of affine transform", "second moment is variance plus squared mean"], "squared mean without variance"),
        ])
    elif archetype == "general_moment":
        mean, var, scale, offset = rng.randrange(8, 70), rng.randrange(17, 310), rng.randrange(2, 8), rng.randrange(2, 25)
        prefix = f"X has mean {mean} and variance {var}. Measurements are transformed to Y={scale}*X+{offset}."
        qs.extend([
            question(story_id, "moment_mean", 0, prefix + " Set up E[Y].",
                f"{scale}*{mean}+{offset}", {}, ["affine expectation"], "variance"),
            question(story_id, "moment_variance", 1, prefix + " Set up Var(Y).",
                f"{scale}**2*{var}", {}, ["affine variance"], "including the offset"),
            question(story_id, "moment", 2, prefix + " Set up E[Y**2].",
                f"{scale}**2*{var}+({scale}*{mean}+{offset})**2",
                {"mean": str(mean), "variance": str(var), "scale": str(scale), "offset": str(offset)},
                ["affine variance", "affine mean", "second moment is variance plus squared mean"], "squared mean without variance"),
        ])
    elif archetype == "uniform_wait":
        number_form = "mixed_seconds_and_minutes"
        upper, seconds = rng.randrange(22, 90), rng.randrange(61, 1000)
        cutoff = f"{seconds}/60"
        qs.extend([
            question(story_id, "uniform_time", 0,
                f"A total wait T is uniform from 0 to {upper} minutes. After already waiting {seconds} seconds, find E[T | T>{seconds} seconds] in minutes.",
                f"({cutoff}+{upper})/2", {"upper_minutes": str(upper), "cutoff_minutes": cutoff},
                ["convert seconds to minutes", "truncate uniform support", "mean is midpoint"], "remaining rather than total wait"),
            question(story_id, "v18_conditional_wait", 1,
                f"For the same wait, set up the expected additional wait after {seconds} seconds have elapsed.",
                f"({upper}-{cutoff})/2", {}, ["conditional total mean", "subtract elapsed time"], "conditional total wait"),
        ])
    elif archetype == "binomial":
        number_form = "percent"
        n, r, pct = rng.randrange(5, 13), rng.randrange(1, 5), rng.randrange(8, 61)
        r = min(r, n - 1)
        qs.append(question(story_id, "binomial", 0,
            f"Among {n} independent quality checks, each flags with probability {pct}%. Set up the probability that exactly {r} checks flag.",
            f"comb({n},{r})*({pct}/100)**{r}*(1-{pct}/100)**({n}-{r})",
            {"n": str(n), "r": str(r), "reject_probability": f"{pct}/100"},
            ["choose the flagged checks", "multiply success and failure probabilities"], "omitting the combination count"))
    elif archetype == "detection_events":
        pa, pb, den = rng.randrange(8, 88), rng.randrange(9, 89), (100 if i % 2 == 0 else 97)
        while pb == pa:
            pb = rng.randrange(9, 89)
        number_form = "percent_denominator_100" if den == 100 else "nondecimal_fraction_denominator_97"
        p, q = f"({pa}/{den})", f"({pb}/{den})"
        stem = f"Independent alarms A and B activate with probabilities {pa}/{den} and {pb}/{den}."
        items = (
            ("exactly_one", "exactly one alarm activates", f"{p}*(1-{q})+(1-{p})*{q}", "the alarms agree"),
            ("at_least_one", "at least one alarm activates", f"1-(1-{p})*(1-{q})", "both alarms"),
            ("both", "both alarms activate", f"{p}*{q}", "at least one alarm"),
            ("neither", "neither alarm activates", f"(1-{p})*(1-{q})", "exactly one alarm"),
            ("same", "the two alarms have the same activation state", f"{p}*{q}+(1-{p})*(1-{q})", "exactly one alarm"),
        )
        for j, (cat, ask, expr, confused) in enumerate(items):
            qs.append(question(story_id, cat, j, stem + " Set up the probability that " + ask + ".",
                expr, {}, ["enumerate independent joint outcomes", cat], confused))
    elif archetype == "interval":
        lower, width, divisor = rng.randrange(20, 130), rng.randrange(18, 190), rng.randrange(2, 8)
        upper = lower + width
        qs.append(question(story_id, "interval", 0,
            f"A normal-theory interval is [{lower}, {upper}]. Sample size is multiplied by {divisor**2}; center, confidence level and population deviation stay fixed. Set up the new upper endpoint.",
            f"({lower}+{upper})/2+({upper}-{lower})/(2*{divisor})",
            {"lower": str(lower), "upper": str(upper), "width_divisor": str(divisor)},
            ["preserve center", "standard error shrinks by square-root sample-size factor"], "dividing the endpoint"))
    elif archetype == "chain_poisson_variance":
        number_form = "mixed_seconds_and_minutes"
        rate, minutes, seconds = rng.randrange(13, 90), rng.randrange(2, 15), rng.randrange(61, 800)
        specs = (
            ("A Poisson count X has mean {r}. Set up Var(X).".format(r=rate), str(rate)),
            (f"A Poisson process runs at {rate} per minute for {minutes} minutes. Set up the count variance.", f"{rate}*{minutes}"),
            (f"A Poisson process runs at {rate} per minute for {seconds} seconds. Set up the count variance.", f"{rate}*({seconds}/60)"),
        )
        for j, (text, expr) in enumerate(specs):
            qs.append(question(story_id, "v18_poisson_variance", j, text, expr, {},
                ["Poisson variance"] if j == 0 else (["rate times duration", "Poisson variance"] if j == 1 else ["unit conversion", "rate times duration", "Poisson variance"]),
                "wrong duration unit"))
    elif archetype == "chain_scaled_variance":
        lam, var, scale, offset, minutes = rng.randrange(12, 90), rng.randrange(19, 300), rng.randrange(2, 8), rng.randrange(3, 28), rng.randrange(2, 12)
        specs = (
            (f"X has variance {var}; Y={scale}*X+{offset}. Set up Var(Y).", f"{scale}**2*{var}"),
            (f"X is Poisson with mean {lam}; Y={scale}*X+{offset}. Set up Var(Y).", f"{scale}**2*{lam}"),
            (f"A Poisson process runs at {lam} per minute for {minutes} minutes. X is its count and Y={scale}*X+{offset}. Set up Var(Y).", f"{scale}**2*({lam}*{minutes})"),
        )
        for j, (text, expr) in enumerate(specs):
            qs.append(question(story_id, "v18_scaled_variance", j, text, expr, {}, ["affine variance"] * (j + 1), "linear rather than squared scale"))
    elif archetype == "chain_second_moment":
        mean, var, minutes = rng.randrange(12, 90), rng.randrange(20, 310), rng.randrange(2, 12)
        specs = (
            (f"X has mean {mean} and variance {var}. Set up E[X**2].", f"{var}+{mean}**2"),
            (f"X is Poisson with mean {mean}. Set up E[X**2].", f"{mean}+{mean}**2"),
            (f"A Poisson process runs at {mean} per minute for {minutes} minutes. X is the total count. Set up E[X**2].", f"({mean}*{minutes})+({mean}*{minutes})**2"),
        )
        for j, (text, expr) in enumerate(specs):
            qs.append(question(story_id, "v18_second_moment", j, text, expr, {}, ["second moment"] * (j + 1), "variance or squared mean alone"))
    else:
        raise ValueError(archetype)
    for q in qs:
        q["question"] = apply_surface(q["question"], style)
    return {
        "story_id": story_id,
        "lineage_id": f"consolidation_{archetype}_{i:03d}",
        "archetype": archetype,
        "split": split,
        "surface_style": style,
        "number_form": number_form,
        "source": "consolidation_candidate_v1",
        "questions": qs,
    }


def build():
    rng = random.Random(SEED)
    stories = []
    for archetype, total, validation_count in ARCHETYPES:
        for i in range(total):
            split = "validation" if i >= total - validation_count else "train"
            stories.append(bundle(archetype, split, i, rng))
    pilot_ids = []
    by_arch = defaultdict(list)
    for s in stories:
        if s["split"] == "train":
            by_arch[s["archetype"]].append(s["story_id"])
    for archetype, _, _ in ARCHETYPES:
        pilot_ids.extend(by_arch[archetype][:2])
    for archetype in ("detection_events", "poisson_process", "affine_poisson", "general_moment"):
        pilot_ids.append(by_arch[archetype][2])
    assert len(pilot_ids) == 24
    requests = []
    for story in stories:
        answer_schema = [{"question_id": q["id"], "expression": "<fully substituted expression>"} for q in story["questions"]]
        requests.append({
            "story_id": story["story_id"],
            "lineage_id": story["lineage_id"],
            "split": story["split"],
            "archetype": story["archetype"],
            "pilot_batch": story["story_id"] in pilot_ids,
            "teacher_model": TEACHER_MODEL,
            "questions": [{"question_id": q["id"], "question": q["question"]} for q in story["questions"]],
            "response_schema": {"answers": answer_schema},
        })
    owners = {}
    for s in stories:
        assert owners.setdefault(s["lineage_id"], s["split"]) == s["split"]
        assert len({q["id"] for q in s["questions"]}) == len(s["questions"])
    counts = Counter(q["category"] for s in stories for q in s["questions"])
    split_counts = {split: Counter(q["category"] for s in stories if s["split"] == split for q in s["questions"])
                    for split in ("train", "validation")}
    coverage = {
        "status": "candidate_not_frozen",
        "seed": SEED,
        "teacher_model": TEACHER_MODEL,
        "story_count": len(stories),
        "question_count": sum(len(s["questions"]) for s in stories),
        "pilot_story_count": len(pilot_ids),
        "stories_by_split": Counter(s["split"] for s in stories),
        "questions_by_category": counts,
        "questions_by_split_and_category": split_counts,
        "coverage_axes": ["concept", "story lineage", "surface style", "number form", "unit", "condition order", "confusable requested quantity"],
        "category_groups": {
            "historical_eight": ["poisson_time", "poisson_scaled", "moment", "uniform_time", "binomial", "exactly_one", "at_least_one", "interval"],
            "combination_four": ["v18_poisson_variance", "v18_scaled_variance", "v18_second_moment", "v18_conditional_wait"],
            "event_four": ["exactly_one", "both", "neither", "same"],
            "supporting_contrasts": ["poisson_scaled_mean", "poisson_scaled_moment", "moment_mean", "moment_variance"],
        },
        "notes": [
            "Counts are target responses, not independent stories.",
            "Train and validation lineages are disjoint before any teacher call.",
            "Concrete new holdout questions are owned by a separate future builder and are absent from teacher requests.",
        ],
    }
    holdout = {
        "status": "blueprint_only_no_concrete_questions",
        "owner": "future_separate_evaluation_builder",
        "forbidden_inputs": ["STATS_CONSOLIDATION_PILOT_REQUESTS.json", "teacher generation prompts", "teacher retry prompts"],
        "planned_counts": {"historical_eight": "8 categories x 12 = 96", "combination_four": "4 categories x 24 = 96", "event_four": "4 categories x 24 = 96", "permanent_mc": "60 questions x 4 rotations = 240 responses"},
        "reserved_axes": ["unseen story domains", "unseen wording families", "condition reordering", "seconds/minutes and fractional minutes", "percent/fraction probability", "different denominators", "confusable requested quantities"],
        "leakage_rule": "No story, lineage, parameter tuple, wording family or concrete answer may enter teacher generation or training.",
    }
    return stories, requests, coverage, holdout


def selection_validation(stories):
    """Small fixed matrix for frequent checkpoint selection; never uses test rows."""
    old = json.loads((DOCS / "STATS_V0_19_FROZEN_QUESTIONS.json").read_text())
    event = json.loads((DOCS / "STATS_V0_20_FROZEN_QUESTIONS.json").read_text())
    suites = {}
    suites["old"] = [q for category in KINDS for q in [x for x in old["old_validation"] if x["category"] == category][:4]]
    chain_categories = ("v18_poisson_variance", "v18_scaled_variance", "v18_second_moment", "v18_conditional_wait")
    suites["chain"] = [q for category in chain_categories for q in [x for x in old["new_validation"] if x["category"] == category][:6]]
    event_categories = ("exactly_one", "both", "neither", "same")
    suites["event"] = [q for category in event_categories for q in [x for x in event["validation"] if x["category"] == category][:4]]
    candidate_pool = [q for s in stories if s["split"] == "validation" for q in s["questions"]]
    categories = sorted({q["category"] for q in candidate_pool})
    suites["candidate"] = [next(q for q in candidate_pool if q["category"] == category) for category in categories]
    assert {k: len(v) for k, v in suites.items()} == {"old": 32, "chain": 24, "event": 16, "candidate": 19}
    return {"status": "candidate_not_frozen", "role": "checkpoint_selection_only",
            "source": "validation splits only; no test rows", "suites": suites}


def main():
    stories, requests, coverage, holdout = build()
    selection = selection_validation(stories)
    outputs = {
        "STATS_CONSOLIDATION_CANDIDATE_STORIES.json": stories,
        "STATS_CONSOLIDATION_PILOT_REQUESTS.json": requests,
        "STATS_CONSOLIDATION_COVERAGE.json": coverage,
        "STATS_CONSOLIDATION_HOLDOUT_BLUEPRINT.json": holdout,
        "STATS_CONSOLIDATION_SELECTION_VALIDATION.json": selection,
    }
    for name, value in outputs.items():
        (DOCS / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"hashes": {k: digest(v) for k, v in outputs.items()}, "coverage": coverage}, default=dict))


if __name__ == "__main__":
    main()
