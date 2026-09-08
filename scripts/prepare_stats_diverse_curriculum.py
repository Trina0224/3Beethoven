"""Build the frozen, coverage-slotted statistics curriculum.

The target expressions in this file are assistant-authored expansions of
independently checked statistical identities.  A cloud teacher is not the
ground-truth authority: each row must pass prompt/semantics binding checks, the
independent rational oracle, and the isolated formulation grader.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter
from fractions import Fraction as F
from pathlib import Path

from exact_calculator import calculate
from stats_consolidation_grader import GRADER_VERSION, grader_fingerprint, reviewed_shape, score
from stats_consolidation_semantics import (
    SEMANTICS_VERSION,
    oracle,
    spec_for,
    task_key,
    validate_question,
)


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DEFAULT_OUTPUT = DOCS / "STATS_DIVERSE_CURRICULUM.json"
DEFAULT_BLIND_OUTPUT = DOCS / "STATS_DIVERSE_FINAL_BLIND.json"
DATA_REVISION = "stats-diverse-curriculum-v2"
PROMPT_PARSER_VERSION = "stats-diverse-direct-parser-v2"
TEMPLATE_NORMALIZER_VERSION = "nfkc-lower-number-mask-whitespace-v1"
SEED = 684031
PROVENANCE = "assistant_authored_verified_rule_expansion"
SUFFIX = (
    "\nReturn one line: Expression: <a fully substituted numerical expression>. "
    "Keep arithmetic and unit conversion operations unevaluated. Use integers, "
    "fractions, +, -, *, /, ** and comb(n,r). No final answer or explanation."
)

# Whole-story contract: train must be divisible by both the five-row binomial
# contrast and the four-row interval contrast; evaluation splits must be
# divisible by the five-row binomial and two-row interval contrasts.
SPLIT_SIZES = {"train": 40, "development": 10, "final_blind": 10}
EVENT_CATEGORIES = ("both", "neither", "exactly_one", "at_least_one", "same")
MOMENT_CATEGORIES = ("moment_mean", "moment_variance", "moment_second")
POISSON_CATEGORIES = ("poisson_variance", "poisson_scaled", "poisson_second")
PROCESS_CATEGORIES = ("process_variance", "process_scaled", "process_second")
UNIFORM_CATEGORIES = ("uniform_mean", "uniform_conditional")
CATEGORIES = (
    *EVENT_CATEGORIES,
    *MOMENT_CATEGORIES,
    *POISSON_CATEGORIES,
    *PROCESS_CATEGORIES,
    *UNIFORM_CATEGORIES,
    "binomial",
    "interval",
)
FAMILY_CATEGORIES = {
    "events": EVENT_CATEGORIES,
    "moments": MOMENT_CATEGORIES,
    "poisson": POISSON_CATEGORIES,
    "process": PROCESS_CATEGORIES,
    "uniform": UNIFORM_CATEGORIES,
    "binomial": ("binomial",),
    "interval": ("interval",),
}
FAMILY_ORDER = tuple(FAMILY_CATEGORIES)
CATEGORY_TO_FAMILY = {
    category: family for family, categories in FAMILY_CATEGORIES.items() for category in categories
}
CYCLE_FAMILY_SCHEDULE = (
    "events", "poisson", "poisson", "process", "moments", "events",
    "process", "events", "moments", "poisson", "interval", "events",
    "uniform", "uniform", "events", "binomial", "process", "moments",
)
HELD_OUT_STRUCTURAL_COMBINATIONS = {
    "event_same_denominator_b_then_a": {
        "family": "events",
        "denominator_relation": "same",
        "given_order": "B_then_A",
        "probability_a_position": "interior",
        "probability_b_position": "interior",
    },
    "moment_all_negative_fraction_affine": {
        "family": "moments",
        "mean_form": "negative_fraction",
        "scale_form": "negative_fraction",
        "offset_form": "negative_fraction",
    },
    "poisson_fractional_mean_negative_fraction_scale_offset": {
        "family": "poisson",
        "concept": "scaled_variance",
        "mean_form": "positive_fraction",
        "scale_form": "negative_fraction",
        "offset_form": "negative_fraction",
    },
    "process_seconds_fractional_rate_negative_fraction_scale": {
        "family": "process",
        "concept": "scaled_variance",
        "duration_unit": "seconds",
        "rate_form": "positive_fraction",
        "scale_form": "negative_fraction",
    },
    "uniform_fractional_lower_seconds_cutoff": {
        "family": "uniform",
        "concept": "conditional_total_mean",
        "lower_form": "positive_fraction",
        "cutoff_unit": "seconds",
    },
    "binomial_even_n_above_half": {
        "family": "binomial",
        "n_parity": "even",
        "probability_region": "above_half",
    },
    "interval_negative_fractional_rational_multiplier": {
        "family": "interval",
        "interval_geometry": "negative",
        "endpoint_form": "contains_fraction",
        "sample_multiplier_form": "positive_fraction",
    },
}

EVENT_ASK = {
    "both": "both A and B occur",
    "neither": "neither A nor B occurs",
    "exactly_one": "exactly one of A and B occurs",
    "at_least_one": "at least one of A and B occurs",
    "same": "both occur or neither occurs",
}
MOMENT_TARGET = {
    "moment_mean": ("mean", "E[Y]"),
    "moment_variance": ("variance", "Var(Y)"),
    "moment_second": ("second_moment", "E[Y**2]"),
}


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def normalize_template(text: str) -> str:
    """Frozen signature normalizer shared by builders and execution gates."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"(?<![\w])[-+]?(?:\d+/\d+|\d+(?:\.\d+)?)(?![\w])", "<num>", text)
    return " ".join(text.split())


def template_signature(question: str, prompt: str) -> str:
    return text_digest(normalize_template(question) + "\n" + normalize_template(prompt))


def canonical_key(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def as_jsonable(value):
    if isinstance(value, tuple):
        return [as_jsonable(item) for item in value]
    if isinstance(value, list):
        return [as_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: as_jsonable(item) for key, item in value.items()}
    return value


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def seeded_rng(*parts) -> random.Random:
    material = "|".join(map(str, (SEED, *parts))).encode()
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))


def qstr(value) -> str:
    """Canonical rational spelling used in semantics and generated expressions."""
    return str(F(value))


def classify_number(value) -> str:
    value = F(value)
    if value == 0:
        return "zero"
    sign = "negative" if value < 0 else "positive"
    form = "fraction" if value.denominator != 1 else "integer"
    return f"{sign}_{form}"


def probability_position(value) -> str:
    value = F(value)
    if value == 0:
        return "zero"
    if value == 1:
        return "one"
    return "interior"


def noninteger_fraction(rng, low_numerator=1, high_numerator=90, denominators=(7, 11, 13, 17, 19)) -> F:
    denominator = rng.choice(denominators)
    numerator = rng.randrange(low_numerator, high_numerator + 1)
    while numerator % denominator == 0:
        numerator += 1
    return F(numerator, denominator)


def json_document(path: Path):
    raw = path.read_bytes()
    if path.name.endswith(".json.gz.b64"):
        decoded = gzip.decompress(base64.b64decode(raw))
        return json.loads(decoded), hashlib.sha256(decoded).hexdigest()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def story_projection(spec):
    """Canonical primitive story, deliberately broader than a target task key."""
    kind = spec["kind"]
    if kind == "events":
        return (kind, *sorted((qstr(spec["p"]), qstr(spec["p_b"]))))
    if kind == "moments":
        return (kind, qstr(spec.get("mean", 0)), qstr(spec.get("variance", 0)))
    if kind == "poisson":
        return (kind, qstr(spec["mean"]))
    if kind == "process":
        return (kind, qstr(spec["rate"]), qstr(spec["duration"]))
    if kind == "uniform":
        return (kind, qstr(spec["lower"]), qstr(spec["upper"]))
    if kind == "binomial":
        return (kind, qstr(spec["n"]), qstr(spec["p"]))
    if kind == "interval":
        return (kind, qstr(spec["lower"]), qstr(spec["upper"]))
    raise ValueError(f"Unknown story family: {kind}")


def display_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def historical_registry(output_paths=(DEFAULT_OUTPUT, DEFAULT_BLIND_OUTPUT)):
    """Scan plain and compressed JSON history, including the final_clear corpus."""
    excluded = {Path(path).resolve() for path in output_paths}
    paths = sorted(set(DOCS.rglob("*.json")) | set(DOCS.rglob("*.json.gz.b64")))
    tasks, projections = set(), set()
    questions, prompts = set(), set()
    source_files = {}
    candidate_records = accepted_records = rejected_records = 0
    for path in paths:
        if path.resolve() in excluded or path.name.startswith("STATS_DIVERSE_"):
            continue
        payload, decoded_sha256 = json_document(path)
        source_files[str(path.relative_to(ROOT))] = {
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "decoded_json_sha256": decoded_sha256,
            "encoding": "gzip_base64" if path.name.endswith(".json.gz.b64") else "json",
        }
        for item in walk(payload):
            if isinstance(item.get("question"), str):
                questions.add(item["question"])
            if isinstance(item.get("prompt"), str):
                prompts.add(item["prompt"])
            if "question" not in item or not ("semantics" in item or "category" in item):
                continue
            candidate_records += 1
            try:
                spec = spec_for(item)
                tasks.add(canonical_key(task_key(item)))
                projections.add(canonical_key(story_projection(spec)))
                accepted_records += 1
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                # Historical result bundles contain model-response dictionaries and
                # partial audit records too. They are counted, never mistaken for
                # validated semantic questions.
                rejected_records += 1
    if not any(name.endswith("STATS_FINAL_CLEAR_DATA.json.gz.b64") for name in source_files):
        raise RuntimeError("Historical scan did not include the final_clear compressed corpus")
    audit = {
        "scanned_globs": ["docs/**/*.json", "docs/**/*.json.gz.b64"],
        "excluded_current_outputs": [
            "docs/STATS_DIVERSE_CURRICULUM.json",
            "docs/STATS_DIVERSE_FINAL_BLIND.json",
        ],
        "excluded_current_experiment_glob": "docs/**/STATS_DIVERSE_*",
        "source_count": len(source_files),
        "source_files": source_files,
        "source_files_sha256": digest(source_files),
        "candidate_question_records": candidate_records,
        "accepted_semantic_records": accepted_records,
        "rejected_partial_records": rejected_records,
        "task_key_count": len(tasks),
        "task_key_registry_sha256": digest(sorted(tasks)),
        "primitive_story_count": len(projections),
        "primitive_story_registry_sha256": digest(sorted(projections)),
        "all_text_question_count": len(questions),
        "exact_question_registry_sha256": digest(sorted(questions)),
        "all_text_prompt_count": len(prompts),
        "exact_prompt_registry_sha256": digest(sorted(prompts)),
        "normalized_question_registry_sha256": digest(
            sorted(text_digest(normalize_template(value)) for value in questions)
        ),
        "normalized_prompt_registry_sha256": digest(
            sorted(text_digest(normalize_template(value)) for value in prompts)
        ),
    }
    return tasks, projections, questions, prompts, audit


def build_question(category, semantics, axes):
    """Render from semantics; validation re-renders instead of trusting shadow bindings."""
    s = semantics
    surface = axes["surface_form"]
    if surface not in ("givens_first", "target_first"):
        raise ValueError("Unknown controlled surface form")
    if category in EVENT_CATEGORIES:
        a, b = s["p"], s["p_b"]
        first = f"P(A)=({a}); P(B)=({b})"
        if axes["given_order"] == "B_then_A":
            first = f"P(B)=({b}); P(A)=({a})"
        if surface == "givens_first":
            return f"A and B are independent events. {first}. Find the probability that {EVENT_ASK[category]}."
        return f"For independent events A and B, find the probability that {EVENT_ASK[category]}. Given {first}."
    if category in MOMENT_CATEGORIES:
        _, ask = MOMENT_TARGET[category]
        facts = f"E[X]=({s['mean']}); Var(X)=({s['variance']}); Y=({s['scale']})*X+({s['offset']})"
        if surface == "givens_first":
            return f"E[X]=({s['mean']}); Var(X)=({s['variance']}). Define Y=({s['scale']})*X+({s['offset']}). Find {ask}."
        return f"Find {ask}. Given {facts}."
    if category in POISSON_CATEGORIES:
        text = f"X has a Poisson distribution with mean ({s['mean']}). "
        if category == "poisson_scaled":
            if surface == "givens_first":
                return text + f"Define Y=({s['scale']})*X+({s['offset']}). Find Var(Y)."
            return f"Find Var(Y). Given X is Poisson with mean ({s['mean']}) and Y=({s['scale']})*X+({s['offset']})."
        ask = "Var(X)" if category == "poisson_variance" else "E[X**2]"
        if surface == "givens_first":
            return text + f"Find {ask}."
        return f"Find {ask}. Given X is Poisson with mean ({s['mean']})."
    if category in PROCESS_CATEGORIES:
        duration = F(s["duration"])
        if axes["duration_unit"] == "seconds":
            seconds = duration * 60
            if seconds.denominator != 1:
                raise ValueError("A seconds prompt must display an integer number of seconds")
            shown_duration, unit = str(seconds.numerator), "seconds"
        else:
            shown_duration, unit = s["duration"], "minutes"
        facts = (
            f"A homogeneous Poisson process has rate ({s['rate']}) arrivals per minute. "
            f"X counts arrivals during ({shown_duration}) {unit}. "
        )
        if category == "process_scaled":
            if surface == "givens_first":
                return facts + f"Define Y=({s['scale']})*X+({s['offset']}). Find Var(Y)."
            return f"Find Var(Y). Given a homogeneous Poisson process with rate ({s['rate']}) arrivals per minute, X counts arrivals during ({shown_duration}) {unit}, and Y=({s['scale']})*X+({s['offset']})."
        ask = "Var(X)" if category == "process_variance" else "E[X**2]"
        if surface == "givens_first":
            return facts + f"Find {ask}."
        return f"Find {ask}. Given a homogeneous Poisson process with rate ({s['rate']}) arrivals per minute, X counts arrivals during ({shown_duration}) {unit}."
    if category in UNIFORM_CATEGORIES:
        text = f"T is uniform on [({s['lower']}), ({s['upper']})] minutes. "
        if category == "uniform_mean":
            if surface == "givens_first":
                return text + "Find E[T] in minutes."
            return f"Find E[T] in minutes. Given T is uniform on [({s['lower']}), ({s['upper']})] minutes."
        cutoff = F(s["cutoff"])
        if axes["cutoff_unit"] == "seconds":
            seconds = cutoff * 60
            if seconds.denominator != 1:
                raise ValueError("A seconds cutoff must be an integer number of seconds")
            shown_cutoff, unit = str(seconds.numerator), "seconds"
        else:
            shown_cutoff, unit = s["cutoff"], "minutes"
        if surface == "givens_first":
            return text + f"Given T>({shown_cutoff}) {unit}, find the conditional mean of total T in minutes."
        return f"Find the conditional mean of total T in minutes, given T>({shown_cutoff}) {unit} and T uniform on [({s['lower']}), ({s['upper']})] minutes."
    if category == "binomial":
        if surface == "givens_first":
            return f"X is the number of successes in ({s['n']}) independent trials, each with success probability ({s['p']}). Find P(X=({s['r']}))."
        return f"Find P(X=({s['r']})). Given X counts successes in ({s['n']}) independent trials with success probability ({s['p']}) per trial."
    if category == "interval":
        multiplier = qstr(F(s["divisor"]) ** 2)
        facts = f"a normal-theory confidence interval is [({s['lower']}), ({s['upper']})], the sample size is multiplied by ({multiplier}), and the center, confidence level, and population standard deviation stay fixed"
        if surface == "givens_first":
            return f"A normal-theory confidence interval is [({s['lower']}), ({s['upper']})]. The sample size is multiplied by ({multiplier}); the center, confidence level, and population standard deviation stay fixed. Find the new upper endpoint."
        return f"Find the new upper endpoint, given {facts}."
    raise ValueError(f"Unknown category: {category}")


def expression_for(category, s, axes):
    p = lambda key: f"({s[key]})"
    if category in EVENT_CATEGORIES:
        a, b = p("p"), p("p_b")
        return {
            "both": f"{a}*{b}",
            "neither": f"(1-{a})*(1-{b})",
            "exactly_one": f"{a}*(1-{b})+(1-{a})*{b}",
            "at_least_one": f"1-(1-{a})*(1-{b})",
            "same": f"{a}*{b}+(1-{a})*(1-{b})",
        }[category]
    if category in MOMENT_CATEGORIES:
        a, m, v, z = (p(name) for name in ("scale", "mean", "variance", "offset"))
        return {
            "moment_mean": f"{a}*{m}+{z}",
            "moment_variance": f"{a}**2*{v}",
            "moment_second": f"{a}**2*{v}+({a}*{m}+{z})**2",
        }[category]
    if category in POISSON_CATEGORIES:
        mean = p("mean")
        if category == "poisson_variance":
            return mean
        if category == "poisson_scaled":
            return f"{p('scale')}**2*{mean}"
        return f"{mean}+{mean}**2"
    if category in PROCESS_CATEGORIES:
        rate, duration = p("rate"), F(s["duration"])
        if axes["duration_unit"] == "seconds":
            seconds = duration * 60
            lam = f"{rate}*(({seconds.numerator})/60)"
        else:
            lam = f"{rate}*{p('duration')}"
        if category == "process_variance":
            return lam
        if category == "process_scaled":
            return f"{p('scale')}**2*({lam})"
        return f"({lam})+({lam})**2"
    if category == "uniform_mean":
        return f"({p('lower')}+{p('upper')})/2"
    if category == "uniform_conditional":
        return f"({p('cutoff')}+{p('upper')})/2"
    if category == "binomial":
        n, r, probability = p("n"), p("r"), p("p")
        return f"comb({n},{r})*{probability}**{r}*(1-{probability})**({n}-{r})"
    if category == "interval":
        lo, hi, divisor = p("lower"), p("upper"), p("divisor")
        return f"({lo}+{hi})/2+({hi}-{lo})/(2*{divisor})"
    raise ValueError(f"Unknown category: {category}")


def prompt_bindings(category, s, axes):
    bindings = {key: value for key, value in s.items() if key not in ("kind", "target")}
    if category in PROCESS_CATEGORIES:
        bindings["display_duration_unit"] = axes["duration_unit"]
        bindings["display_duration"] = (
            str((F(s["duration"]) * 60).numerator)
            if axes["duration_unit"] == "seconds"
            else s["duration"]
        )
    if category == "uniform_conditional":
        bindings["display_cutoff_unit"] = axes["cutoff_unit"]
        bindings["display_cutoff"] = (
            str((F(s["cutoff"]) * 60).numerator)
            if axes["cutoff_unit"] == "seconds"
            else s["cutoff"]
        )
    if category == "interval":
        bindings["sample_size_multiplier"] = qstr(F(s["divisor"]) ** 2)
    return bindings


def make_row(split, category, index, family, semantics, axes, group_key, contrast_size):
    axes = dict(axes)
    axes["surface_form"] = "givens_first" if index % 2 == 0 else "target_first"
    question = build_question(category, semantics, axes)
    expression = expression_for(category, semantics, axes)
    group_story_id = "diverse_" + digest(group_key)[:24]
    row_id = f"diverse_{split}_{category}_{index:03d}"
    target = "Expression: " + expression
    signature = template_signature(question, question + SUFFIX)
    stratum = ";".join(
        f"{key}={axes[key]}"
        for key in sorted(axes)
        if key not in ("family", "contrast_design")
    )
    row = {
        "id": row_id,
        "split": split,
        "category": category,
        "family": family,
        "question": question,
        "prompt": question + SUFFIX,
        "prompt_bindings": prompt_bindings(category, semantics, axes),
        "semantics": semantics,
        "expression": expression,
        "answer": str(oracle(semantics)),
        "target": target,
        "bindings": (
            {
                "lower": semantics["lower"],
                "upper": semantics["upper"],
                "width_divisor": semantics["divisor"],
            }
            if category == "interval"
            else {}
        ),
        "semantic_key": as_jsonable(task_key({"id": "draft", "category": category, "semantics": semantics})),
        "group_story_key": as_jsonable(group_key),
        "group_story_id": group_story_id,
        "story_id": group_story_id,
        "contrast_group": group_story_id,
        "contrast_size": contrast_size,
        "coverage_axes": axes,
        "stratum": stratum,
        "template_signature": signature,
        "misconception_traps": [],
        "discriminative_mutations": [],
        "provenance": PROVENANCE,
        "target_authority": "canonical_symbolic_oracle",
        "teacher_eligible": split == "train",
        "teacher_exposed": False,
        "training_eligible": split == "train",
        "teacher_binding": (
            {
                "result_id": f"teacher_result_{row_id}",
                "training_row_id": row_id,
                "request_prompt_sha256": text_digest(question + SUFFIX),
                "canonical_target_sha256": text_digest(target),
                "collection_status": "pending_external_train_only_collection",
                "required_release_state": "accepted_or_oracle_corrected",
                "required_review_required": False,
            }
            if split == "train"
            else None
        ),
    }
    return row


def event_candidate(split, index, attempt):
    rng = seeded_rng(split, "events", index, attempt)
    denominators = (107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163)
    offset = (index * 3 + attempt * 5 + {"train": 0, "development": 2, "final_blind": 4}[split]) % len(denominators)
    den_a = denominators[offset]
    if index % 3 == 0:
        den_b = den_a
        denominator_relation = "same"
    else:
        den_b = denominators[(offset + 1 + index % 4) % len(denominators)]
        if den_a == den_b:
            den_b = denominators[(offset + 3) % len(denominators)]
        denominator_relation = "different"
    num_a = rng.randrange(1, den_a)
    num_b = rng.randrange(1, den_b)
    if split == "train" and index == 0:
        num_a = 0
    elif split == "train" and index == 2:
        num_a = den_a
    elif split == "train" and index % 13 == 1:
        num_b = 0
    p_a, p_b = f"{num_a}/{den_a}", f"{num_b}/{den_b}"
    if split != "train":
        left, right = F(p_a), F(p_b)
        if not (0 < left < 1 and 0 < right < 1) or left == right or left + right == 1:
            raise ValueError("Degenerate event parameters are train-only")
    # Final index zero crosses two already exposed axes: same denominators and
    # reverse presentation order.  Train/development never contain that
    # conjunction, while still covering each axis independently.
    if split == "final_blind" and index == 0:
        order = "B_then_A"
    elif denominator_relation == "same":
        order = "A_then_B"
    else:
        order = "A_then_B" if index % 2 == 0 else "B_then_A"
    group_key = ("events", qstr(p_a), qstr(p_b), order)
    rows = []
    for category in EVENT_CATEGORIES:
        semantics = {"kind": "events", "target": category, "p": p_a, "p_b": p_b}
        axes = {
            "family": "events",
            "concept": category,
            "probability_a_form": classify_number(p_a),
            "probability_b_form": classify_number(p_b),
            "probability_a_position": probability_position(p_a),
            "probability_b_position": probability_position(p_b),
            "denominator_relation": denominator_relation,
            "given_order": order,
            "contrast_design": "five_correct_concepts_same_story",
        }
        rows.append(make_row(split, category, index, "events", semantics, axes, group_key, 5))
    return rows, group_key


def signed_profile(rng, slot, magnitude_base=2):
    if slot == 0:
        return F(-(magnitude_base + rng.randrange(1, 10)))
    if slot == 1:
        return F(0)
    if slot == 2:
        return noninteger_fraction(rng, 2, 80)
    if slot == 3:
        return -noninteger_fraction(rng, 2, 80)
    return F(magnitude_base + rng.randrange(1, 20))


def moment_candidate(split, index, attempt):
    rng = seeded_rng(split, "moments", index, attempt)
    mean = signed_profile(rng, index % 5, 3)
    variance = (
        F(0)
        if split == "train" and index == 0
        else noninteger_fraction(rng, 3, 140, (11, 13, 17, 19, 23))
        if index % 3 == 1
        else F(2 + rng.randrange(1, 90))
    )
    scale = signed_profile(rng, (index + 3) % 5, 1)
    offset = signed_profile(rng, (index + 1) % 5, 2)
    if split != "train":
        if mean == 0:
            mean = -noninteger_fraction(rng, 3, 50, (7, 11, 13))
        if scale == 0:
            scale = -noninteger_fraction(rng, 2, 30, (7, 11, 13))
        if offset == 0:
            offset = noninteger_fraction(rng, 2, 40, (7, 11, 13))
    if split == "final_blind" and index == 0:
        mean = -noninteger_fraction(rng, 3, 50, (7, 11, 13))
        scale = -noninteger_fraction(rng, 2, 30, (7, 11, 13))
        offset = -noninteger_fraction(rng, 2, 40, (17, 19, 23))
    values = tuple(map(qstr, (mean, variance, scale, offset)))
    group_key = ("moments", *values)
    rows = []
    for category in MOMENT_CATEGORIES:
        target, _ = MOMENT_TARGET[category]
        semantics = {
            "kind": "moments",
            "target": target,
            "mean": values[0],
            "variance": values[1],
            "scale": values[2],
            "offset": values[3],
        }
        axes = {
            "family": "moments",
            "concept": target,
            "mean_form": classify_number(mean),
            "variance_form": classify_number(variance),
            "scale_form": classify_number(scale),
            "offset_form": classify_number(offset),
            "contrast_design": "mean_variance_second_moment_same_affine_story",
        }
        rows.append(make_row(split, category, index, "moments", semantics, axes, group_key, 3))
    return rows, group_key


def poisson_candidate(split, index, attempt):
    rng = seeded_rng(split, "poisson", index, attempt)
    if split == "train" and index == 0:
        mean = F(0)
    elif index % 4 == 1:
        mean = F(701 + {"train": 0, "development": 100, "final_blind": 200}[split] + index + attempt * 17)
    else:
        mean = noninteger_fraction(rng, 5, 250, (17, 19, 23, 29, 31))
    scale = signed_profile(rng, (index + 2) % 5, 1)
    offset = signed_profile(rng, (index + 4) % 5, 1)
    if split != "train":
        if scale == 0:
            scale = -noninteger_fraction(rng, 2, 30, (7, 11, 13))
        if offset == 0:
            offset = noninteger_fraction(rng, 2, 40, (7, 11, 13))
    if split == "final_blind" and index == 0:
        scale = -noninteger_fraction(rng, 2, 30, (7, 11, 13))
        offset = -noninteger_fraction(rng, 2, 40, (17, 19, 23))
    elif split == "development" and classify_number(scale) == classify_number(offset) == "negative_fraction":
        offset = noninteger_fraction(rng, 2, 40, (17, 19, 23))
    mean_s, scale_s, offset_s = map(qstr, (mean, scale, offset))
    group_key = ("poisson", mean_s, scale_s, offset_s)
    specifications = {
        "poisson_variance": {"kind": "poisson", "target": "variance", "mean": mean_s},
        "poisson_scaled": {
            "kind": "poisson",
            "target": "variance",
            "mean": mean_s,
            "scale": scale_s,
            "offset": offset_s,
        },
        "poisson_second": {"kind": "poisson", "target": "second_moment", "mean": mean_s},
    }
    rows = []
    for category in POISSON_CATEGORIES:
        axes = {
            "family": "poisson",
            "concept": specifications[category]["target"] if category != "poisson_scaled" else "scaled_variance",
            "mean_form": classify_number(mean),
            "contrast_design": "variance_scaled_variance_second_moment_same_count_story",
        }
        if category == "poisson_scaled":
            axes.update(scale_form=classify_number(scale), offset_form=classify_number(offset))
        rows.append(make_row(split, category, index, "poisson", specifications[category], axes, group_key, 3))
    return rows, group_key


def process_candidate(split, index, attempt):
    rng = seeded_rng(split, "process", index, attempt)
    if split == "train" and index == 0:
        rate = F(0)
    elif index % 2 == 0:
        rate = noninteger_fraction(rng, 3, 100, (7, 11, 13, 17))
    else:
        rate = F(3 + rng.randrange(1, 60))
    duration_mode = ("seconds", "minutes_integer", "minutes_fraction")[index % 3]
    if duration_mode == "seconds":
        seconds = 71 + rng.randrange(0, 700)
        while seconds % 60 == 0:
            seconds += 1
        duration = F(seconds, 60)
        duration_unit = "seconds"
    elif duration_mode == "minutes_integer":
        duration = F(2 + rng.randrange(1, 15))
        duration_unit = "minutes"
    else:
        duration = noninteger_fraction(rng, 4, 60, (3, 5, 7, 11))
        duration_unit = "minutes"
    if split in ("train", "development") and duration_unit == "seconds" and rate.denominator != 1:
        rate = F(5 + rng.randrange(1, 60))
    scale = signed_profile(rng, (index + 1) % 5, 1)
    offset = signed_profile(rng, (index + 3) % 5, 1)
    if split != "train" and scale == 0:
        scale = -noninteger_fraction(rng, 2, 30, (7, 11, 13))
    rate_s, duration_s, scale_s, offset_s = map(qstr, (rate, duration, scale, offset))
    group_key = ("process", rate_s, duration_s, scale_s, offset_s, duration_unit)
    specifications = {
        "process_variance": {
            "kind": "process",
            "target": "variance",
            "rate": rate_s,
            "duration": duration_s,
        },
        "process_scaled": {
            "kind": "process",
            "target": "variance",
            "rate": rate_s,
            "duration": duration_s,
            "scale": scale_s,
            "offset": offset_s,
        },
        "process_second": {
            "kind": "process",
            "target": "second_moment",
            "rate": rate_s,
            "duration": duration_s,
        },
    }
    rows = []
    for category in PROCESS_CATEGORIES:
        axes = {
            "family": "process",
            "concept": specifications[category]["target"] if category != "process_scaled" else "scaled_variance",
            "rate_form": classify_number(rate),
            "duration_unit": duration_unit,
            "duration_form": classify_number(duration),
            "contrast_design": "variance_scaled_variance_second_moment_same_process_story",
        }
        if category == "process_scaled":
            axes["scale_form"] = classify_number(scale)
        rows.append(make_row(split, category, index, "process", specifications[category], axes, group_key, 3))
    return rows, group_key


def uniform_candidate(split, index, attempt):
    rng = seeded_rng(split, "uniform", index, attempt)
    if index % 8 == 0 and split == "train":
        lower = F(0)
    elif (split == "train" and index % 8 == 4) or (split == "development" and index == 0) or index % 4 == 1:
        lower = F(1 + rng.randrange(1, 12))
    else:
        lower = noninteger_fraction(rng, 1, 40, (3, 5, 7, 11))
    width = (
        noninteger_fraction(rng, 120, 1200, (17, 19, 23, 29, 31))
        if lower == 0 or index % 3 == 2
        else F(8 + rng.randrange(1, 80))
    )
    upper = lower + width
    minimum_second = (lower * 60).numerator // (lower * 60).denominator + 1
    maximum_second = (upper * 60).numerator // (upper * 60).denominator - 1
    if minimum_second >= maximum_second:
        raise ValueError("Uniform support too narrow for an interior seconds cutoff")
    cutoff_seconds = rng.randrange(minimum_second, maximum_second + 1)
    cutoff = F(cutoff_seconds, 60)
    lower_s, upper_s, cutoff_s = map(qstr, (lower, upper, cutoff))
    group_key = ("uniform", lower_s, upper_s, cutoff_s)
    specifications = {
        "uniform_mean": {
            "kind": "uniform",
            "conditional": False,
            "lower": lower_s,
            "cutoff": lower_s,
            "upper": upper_s,
        },
        "uniform_conditional": {
            "kind": "uniform",
            "conditional": True,
            "lower": lower_s,
            "cutoff": cutoff_s,
            "upper": upper_s,
        },
    }
    rows = []
    for category in UNIFORM_CATEGORIES:
        cutoff_unit = "seconds" if index % 2 == 0 else "minutes"
        if split != "final_blind" and lower.denominator != 1:
            cutoff_unit = "minutes"
        axes = {
            "family": "uniform",
            "concept": "unconditional_mean" if category == "uniform_mean" else "conditional_total_mean",
            "lower_form": classify_number(lower),
            "upper_form": classify_number(upper),
            "cutoff_unit": "not_applicable" if category == "uniform_mean" else cutoff_unit,
            "conditionality": "unconditional" if category == "uniform_mean" else "conditional_total",
            "contrast_design": "unconditional_and_conditional_total_same_support",
        }
        rows.append(make_row(split, category, index, "uniform", specifications[category], axes, group_key, 2))
    return rows, group_key


def binomial_group_candidate(split, group_index, indices, attempt):
    rng = seeded_rng(split, "binomial", group_index, attempt)
    # The bounded calculator caps exponents at 32. Story uniqueness comes from
    # the exact probability as well as n, so n need not encode the split.
    n = 8 + (group_index % 15)
    # Final group zero is the predeclared even-n/high-p holdout and contains
    # all five r slots.  Train/development expose even n with low p and odd n
    # with both regions, but never the held-out conjunction.
    high_probability = (
        split == "final_blind" and group_index == 0
    ) or (split != "final_blind" and n % 2 == 1 and group_index % 4 == 1)
    denominator = rng.choice((101, 103, 107, 109, 113))
    if high_probability:
        numerator = rng.randrange(denominator // 2 + 1, denominator)
        probability_region = "above_half"
    else:
        numerator = rng.randrange(2, denominator // 2)
        probability_region = "below_half"
    p = F(numerator, denominator)
    group_key = ("binomial", str(n), qstr(p))
    slot_names = ("r_zero", "r_one", "r_interior", "r_n_minus_one", "r_n")
    rows = []
    for index in indices:
        slot = index % 5
        r = (0, 1, 2 + (group_index % (n - 3)), n - 1, n)[slot]
        semantics = {"kind": "binomial", "n": str(n), "r": str(r), "p": qstr(p)}
        axes = {
            "family": "binomial",
            "concept": "exact_count_probability",
            "r_slot": slot_names[slot],
            "probability_form": classify_number(p),
            "n_parity": "even" if n % 2 == 0 else "odd",
            "probability_region": probability_region,
            "contrast_design": "boundary_and_interior_r_same_trial_story",
        }
        rows.append(
            make_row(split, "binomial", index, "binomial", semantics, axes, group_key, len(indices))
        )
    return rows, group_key


def interval_group_candidate(split, group_index, indices, attempt):
    rng = seeded_rng(split, "interval", group_index, attempt)
    geometry = ("negative", "cross_zero", "positive", "zero_lower")[group_index % 4]
    # Negative+fractional is a predeclared final-only conjunction.  The other
    # geometries still cross endpoint forms in train so geometry and number
    # form are not generally confounded.
    if split == "train":
        integer_endpoints = geometry == "negative" or (group_index // 4) % 2 == 0
    elif split == "development":
        integer_endpoints = True if geometry == "negative" else group_index % 2 == 0
    else:
        integer_endpoints = False if group_index == 0 else group_index % 2 == 0
    if geometry == "negative":
        upper = -F(2 + rng.randrange(1, 30)) if integer_endpoints else -noninteger_fraction(rng, 2, 30, (5, 7, 11))
        lower = upper - F(5 + rng.randrange(1, 25))
    elif geometry == "cross_zero":
        if integer_endpoints:
            lower, upper = -F(2 + rng.randrange(1, 30)), F(2 + rng.randrange(1, 35))
        else:
            lower = -noninteger_fraction(rng, 2, 40, (5, 7, 11))
            upper = noninteger_fraction(rng, 2, 50, (5, 7, 11))
    elif geometry == "zero_lower":
        lower = F(0)
        upper = F(8 + rng.randrange(1, 40)) if integer_endpoints else noninteger_fraction(rng, 10, 80, (3, 5, 7))
    else:
        lower = F(1 + rng.randrange(1, 20)) if integer_endpoints else noninteger_fraction(rng, 1, 25, (5, 7, 11))
        upper = lower + F(6 + rng.randrange(1, 30))
    lower_s, upper_s = map(qstr, (lower, upper))
    group_key = ("interval", lower_s, upper_s)
    divisors = (F(2), F(3, 2), F(5, 3), F(3))
    rows = []
    for index in indices:
        divisor = divisors[index % 4]
        multiplier = divisor**2
        semantics = {
            "kind": "interval",
            "lower": lower_s,
            "upper": upper_s,
            "divisor": qstr(divisor),
        }
        axes = {
            "family": "interval",
            "concept": "new_upper_endpoint",
            "interval_geometry": geometry,
            "endpoint_form": (
                "contains_fraction"
                if lower.denominator != 1 or upper.denominator != 1
                else "integers"
            ),
            "sample_multiplier_form": classify_number(multiplier),
            "width_divisor_form": classify_number(divisor),
            "contrast_design": "multiple_sample_multipliers_same_original_interval",
        }
        rows.append(make_row(split, "interval", index, "interval", semantics, axes, group_key, len(indices)))
    return rows, group_key


def primitive_keys(rows):
    return {canonical_key(story_projection(spec_for(row))) for row in rows}


def evaluation_candidate_is_discriminative(rows, split):
    if split == "train":
        return True
    event_rows = [row for row in rows if row["category"] in EVENT_CATEGORIES]
    if event_rows:
        p, p_b = (F(event_rows[0]["semantics"][key]) for key in ("p", "p_b"))
        if not (0 < p < 1 and 0 < p_b < 1) or p == p_b or p + p_b == 1:
            return False
        if len({F(row["answer"]) for row in event_rows}) != len(EVENT_CATEGORIES):
            return False
    for row in rows:
        for trap in expected_misconception_audit(row):
            if not trap["collides_with_oracle"]:
                continue
            boundary_comb_identity = (
                row["category"] == "binomial"
                and trap["id"] == "omit_binomial_coefficient"
                and F(row["semantics"]["r"]) in (0, F(row["semantics"]["n"]))
            )
            if not boundary_comb_identity:
                return False
    return True


def accept_candidate(
    rows,
    group_key,
    split,
    historical_tasks,
    historical_projections,
    historical_questions,
    historical_prompts,
    generated_tasks,
    generated_projections,
    generated_groups,
    generated_questions,
    generated_prompts,
):
    if not evaluation_candidate_is_discriminative(rows, split):
        return False
    task_strings = [canonical_key(task_key(row)) for row in rows]
    if len(task_strings) != len(set(task_strings)):
        return False
    if any(key in historical_tasks or key in generated_tasks for key in task_strings):
        return False
    questions = [row["question"] for row in rows]
    prompts = [row["prompt"] for row in rows]
    if len(questions) != len(set(questions)) or len(prompts) != len(set(prompts)):
        return False
    if set(questions) & (historical_questions | generated_questions):
        return False
    if set(prompts) & (historical_prompts | generated_prompts):
        return False
    projections = primitive_keys(rows)
    if projections & historical_projections:
        return False
    owner = (split, canonical_key(group_key))
    if any(key in generated_projections and generated_projections[key] != owner for key in projections):
        return False
    group_string = canonical_key(group_key)
    if group_string in generated_groups and generated_groups[group_string] != split:
        return False
    generated_tasks.update(task_strings)
    generated_projections.update({key: owner for key in projections})
    generated_groups[group_string] = split
    generated_questions.update(questions)
    generated_prompts.update(prompts)
    return True


def misconception_candidates(row):
    s, category = row["semantics"], row["category"]
    p = lambda key: f"({s[key]})"
    if category in EVENT_CATEGORIES:
        a, b = p("p"), p("p_b")
        candidates = {
            "both": {
                "use_neither_formula_for_both": f"(1-{a})*(1-{b})",
                "use_union_formula_for_both": f"{a}+{b}-{a}*{b}",
            },
            "neither": {
                "use_both_formula_for_neither": f"{a}*{b}",
                "subtract_probabilities_for_neither": f"1-{a}-{b}",
            },
            "exactly_one": {
                "one_exactly_one_branch_only": f"{a}*(1-{b})",
                "use_union_formula_for_exactly_one": f"{a}+{b}-{a}*{b}",
            },
            "at_least_one": {
                "union_without_subtracting_intersection": f"{a}+{b}",
                "use_exactly_one_formula_for_union": f"{a}*(1-{b})+(1-{a})*{b}",
            },
            "same": {
                "use_both_only_for_same": f"{a}*{b}",
                "use_neither_only_for_same": f"(1-{a})*(1-{b})",
            },
        }[category]
    elif category == "moment_mean":
        a, m, v, z = (p(key) for key in ("scale", "mean", "variance", "offset"))
        candidates = {
            "forget_affine_offset_in_mean": f"{a}*{m}",
            "forget_affine_scale_in_mean": f"{m}+{z}",
        }
    elif category == "moment_variance":
        a, v, z = (p(key) for key in ("scale", "variance", "offset"))
        candidates = {
            "use_scale_not_scale_squared_for_variance": f"{a}*{v}",
            "add_offset_to_variance": f"{a}**2*{v}+{z}",
        }
    elif category == "moment_second":
        a, m, v, z = (p(key) for key in ("scale", "mean", "variance", "offset"))
        candidates = {
            "second_moment_without_variance_term": f"({a}*{m}+{z})**2",
            "return_variance_for_second_moment": f"{a}**2*{v}",
        }
    elif category == "poisson_variance":
        mean = p("mean")
        candidates = {
            "poisson_variance_as_mean_squared": f"{mean}**2",
            "add_one_to_poisson_variance": f"{mean}+1",
        }
    elif category == "poisson_scaled":
        mean, scale, offset = (p(key) for key in ("mean", "scale", "offset"))
        candidates = {
            "use_scale_not_scale_squared": f"{scale}*{mean}",
            "add_offset_to_variance": f"{scale}**2*{mean}+{offset}",
            "forget_poisson_scale": mean,
        }
    elif category == "poisson_second":
        mean = p("mean")
        candidates = {
            "second_moment_as_mean_squared_only": f"{mean}**2",
            "return_variance_for_second_moment": mean,
        }
    elif category == "process_variance":
        rate, duration = p("rate"), p("duration")
        lam = f"{rate}*{duration}"
        candidates = {
            "forget_process_duration": rate,
            "return_lambda_squared_for_process_variance": f"({lam})**2",
        }
        if row["coverage_axes"]["duration_unit"] == "seconds":
            seconds = (F(s["duration"]) * 60).numerator
            candidates["treat_seconds_as_minutes"] = f"{rate}*({seconds})"
    elif category == "process_scaled":
        rate, duration, scale = (p(key) for key in ("rate", "duration", "scale"))
        lam = f"{rate}*{duration}"
        candidates = {
            "use_scale_not_scale_squared": f"{scale}*({lam})",
            "forget_process_duration": f"{scale}**2*{rate}",
        }
    elif category == "process_second":
        rate, duration = p("rate"), p("duration")
        lam = f"{rate}*{duration}"
        candidates = {
            "second_moment_as_lambda_squared_only": f"({lam})**2",
            "return_variance_for_second_moment": lam,
        }
    elif category == "uniform_mean":
        lo, hi, cut = p("lower"), p("upper"), p("cutoff")
        candidates = {
            "use_half_width_instead_of_total_mean": f"({hi}-{lo})/2",
            "use_upper_endpoint_as_mean": hi,
        }
    elif category == "uniform_conditional":
        lo, hi, cut = p("lower"), p("upper"), p("cutoff")
        candidates = {
            "ignore_conditioning_cutoff": f"({lo}+{hi})/2",
            "return_expected_remaining_wait_not_total": f"({hi}-{cut})/2",
        }
    elif category == "binomial":
        n, r, probability = p("n"), p("r"), p("p")
        candidates = {
            "omit_binomial_coefficient": f"{probability}**{r}*(1-{probability})**({n}-{r})",
            "return_expected_success_count": f"{n}*{probability}",
        }
    elif category == "interval":
        lo, hi, divisor = p("lower"), p("upper"), p("divisor")
        multiplier = f"{divisor}**2"
        candidates = {
            "divide_half_width_by_sample_multiplier": f"({lo}+{hi})/2+({hi}-{lo})/(2*({multiplier}))",
            "divide_endpoint_by_width_divisor": f"{hi}/{divisor}",
        }
    else:
        raise ValueError(category)
    # Do not label the correct formula as a trap for categories whose alternate
    # concept happens to be the requested concept.
    return {key: value for key, value in candidates.items() if value != row["expression"]}


def expected_misconception_audit(row):
    expected = F(row["answer"])
    reference_shape = reviewed_shape(row["expression"])
    traps = []
    for trap_id, expression in misconception_candidates(row).items():
        computed = F(calculate(expression))
        collision = computed == expected
        structure_differs = reviewed_shape(expression) != reference_shape
        judged = score("Expression: " + expression, row)
        if not collision and judged["primary_correct"]:
            raise ValueError(("Distinct-value misconception received grader credit", row["id"], trap_id))
        traps.append(
            {
                "id": trap_id,
                "expression": expression,
                "computed": str(computed),
                "collides_with_oracle": collision,
                "structure_differs_from_reference": structure_differs,
                "grader_primary_correct": judged["primary_correct"],
                "audit_status": (
                    "degenerate_numeric_collision_not_a_negative_target"
                    if collision
                    else "distinct_wrong_value"
                ),
            }
        )
    if not traps or all(trap["collides_with_oracle"] for trap in traps):
        expression = f"({row['expression']})+1"
        judged = score("Expression: " + expression, row)
        traps.append(
            {
                "id": "unsupported_plus_one_sanity_mutation",
                "expression": expression,
                "computed": calculate(expression),
                "collides_with_oracle": False,
                "structure_differs_from_reference": True,
                "grader_primary_correct": judged["primary_correct"],
                "audit_status": "distinct_wrong_value",
            }
        )
    return traps


def attach_misconception_audit(row):
    traps = expected_misconception_audit(row)
    row["misconception_traps"] = traps
    row["discriminative_mutations"] = [
        {
            "id": trap["id"],
            "expression": trap["expression"],
            "computed": trap["computed"],
            "structure_differs_from_reference": trap["structure_differs_from_reference"],
            "grader_primary_correct": trap["grader_primary_correct"],
        }
        for trap in traps
        if not trap["collides_with_oracle"]
        and trap["structure_differs_from_reference"]
        and not trap["grader_primary_correct"]
    ]


NUM_PATTERN = r"-?\d+(?:/\d+)?"
CONTROLLED_SURFACE_FORMS = ("givens_first", "target_first")
PARSED_EVENT_PHRASES = {
    "both": "both A and B occur",
    "neither": "neither A nor B occurs",
    "exactly_one": "exactly one of A and B occurs",
    "at_least_one": "at least one of A and B occurs",
    "same": "both occur or neither occurs",
}
PARSED_MOMENT_ASK = {
    "moment_mean": ("mean", "E[Y]"),
    "moment_variance": ("variance", "Var(Y)"),
    "moment_second": ("second_moment", "E[Y**2]"),
}


def rational_sqrt(value):
    value = F(value)
    numerator, denominator = math.isqrt(value.numerator), math.isqrt(value.denominator)
    if numerator * numerator != value.numerator or denominator * denominator != value.denominator:
        raise ValueError("Sample-size multiplier has no exact rational square root")
    return F(numerator, denominator)


def canonical_semantics(spec):
    result = {}
    for key, value in spec.items():
        if key in ("kind", "target") or isinstance(value, bool):
            result[key] = value
        else:
            result[key] = qstr(value)
    return result


def parse_question_semantics(question, category, surface_form):
    """Independent, closed-form parser for the controlled direct interface."""
    if surface_form not in CONTROLLED_SURFACE_FORMS:
        raise ValueError("Unexpected controlled surface form")
    body = question
    num = NUM_PATTERN
    if category in EVENT_CATEGORIES:
        if surface_form == "givens_first":
            pattern = (
                rf"A and B are independent events\. P\((?P<label1>[AB])\)=\((?P<v1>{num})\); "
                rf"P\((?P<label2>[AB])\)=\((?P<v2>{num})\)\. Find the probability that "
                + re.escape(PARSED_EVENT_PHRASES[category]) + r"\."
            )
        else:
            pattern = (
                r"For independent events A and B, find the probability that "
                + re.escape(PARSED_EVENT_PHRASES[category])
                + rf"\. Given P\((?P<label1>[AB])\)=\((?P<v1>{num})\); P\((?P<label2>[AB])\)=\((?P<v2>{num})\)\."
            )
        match = re.fullmatch(pattern, body)
        if not match or match["label1"] == match["label2"]:
            raise ValueError("Event prompt does not match the frozen direct grammar")
        values = {match["label1"]: match["v1"], match["label2"]: match["v2"]}
        return {
            "kind": "events",
            "target": category,
            "p": values["A"],
            "p_b": values["B"],
        }
    if category in MOMENT_CATEGORIES:
        target, ask = PARSED_MOMENT_ASK[category]
        pattern = (
            rf"E\[X\]=\((?P<mean>{num})\); Var\(X\)=\((?P<variance>{num})\)\. Define Y=\((?P<scale>{num})\)\*X\+\((?P<offset>{num})\)\. Find {re.escape(ask)}\."
            if surface_form == "givens_first"
            else rf"Find {re.escape(ask)}\. Given E\[X\]=\((?P<mean>{num})\); Var\(X\)=\((?P<variance>{num})\); Y=\((?P<scale>{num})\)\*X\+\((?P<offset>{num})\)\."
        )
        match = re.fullmatch(pattern, body)
        if not match:
            raise ValueError("Moment prompt does not match the frozen direct grammar")
        return {"kind": "moments", "target": target, **match.groupdict()}
    if category in POISSON_CATEGORIES:
        if category == "poisson_scaled":
            pattern = (
                rf"X has a Poisson distribution with mean \((?P<mean>{num})\)\. Define Y=\((?P<scale>{num})\)\*X\+\((?P<offset>{num})\)\. Find Var\(Y\)\."
                if surface_form == "givens_first"
                else rf"Find Var\(Y\)\. Given X is Poisson with mean \((?P<mean>{num})\) and Y=\((?P<scale>{num})\)\*X\+\((?P<offset>{num})\)\."
            )
            match = re.fullmatch(pattern, body)
            if not match:
                raise ValueError("Scaled Poisson prompt does not match the frozen direct grammar")
            return {"kind": "poisson", "target": "variance", **match.groupdict()}
        ask = r"Var\(X\)" if category == "poisson_variance" else r"E\[X\*\*2\]"
        pattern = (
            rf"X has a Poisson distribution with mean \((?P<mean>{num})\)\. Find {ask}\."
            if surface_form == "givens_first"
            else rf"Find {ask}\. Given X is Poisson with mean \((?P<mean>{num})\)\."
        )
        match = re.fullmatch(pattern, body)
        if not match:
            raise ValueError("Poisson prompt does not match the frozen direct grammar")
        return {
            "kind": "poisson",
            "target": "variance" if category == "poisson_variance" else "second_moment",
            "mean": match["mean"],
        }
    if category in PROCESS_CATEGORIES:
        if surface_form == "givens_first":
            ending = {
                "process_variance": r"Find Var\(X\)\.",
                "process_scaled": rf"Define Y=\((?P<scale>{num})\)\*X\+\((?P<offset>{num})\)\. Find Var\(Y\)\.",
                "process_second": r"Find E\[X\*\*2\]\.",
            }[category]
            pattern = rf"A homogeneous Poisson process has rate \((?P<rate>{num})\) arrivals per minute\. X counts arrivals during \((?P<duration>{num})\) (?P<unit>seconds|minutes)\. {ending}"
        else:
            ask = {
                "process_variance": r"Var\(X\)",
                "process_scaled": r"Var\(Y\)",
                "process_second": r"E\[X\*\*2\]",
            }[category]
            tail = (
                rf", and Y=\((?P<scale>{num})\)\*X\+\((?P<offset>{num})\)\."
                if category == "process_scaled" else r"\."
            )
            separator = ", " if category == "process_scaled" else ", "
            pattern = rf"Find {ask}\. Given a homogeneous Poisson process with rate \((?P<rate>{num})\) arrivals per minute{separator}X counts arrivals during \((?P<duration>{num})\) (?P<unit>seconds|minutes){tail}"
        match = re.fullmatch(pattern, body)
        if not match:
            raise ValueError(f"Process prompt does not match the frozen direct grammar: {surface_form}: {body}")
        duration = F(match["duration"]) / 60 if match["unit"] == "seconds" else F(match["duration"])
        parsed = {
            "kind": "process",
            "target": "second_moment" if category == "process_second" else "variance",
            "rate": match["rate"],
            "duration": qstr(duration),
        }
        if category == "process_scaled":
            parsed.update(scale=match["scale"], offset=match["offset"])
        return parsed
    if category in UNIFORM_CATEGORIES:
        if category == "uniform_mean":
            pattern = (
                rf"T is uniform on \[\((?P<lower>{num})\), \((?P<upper>{num})\)\] minutes\. Find E\[T\] in minutes\."
                if surface_form == "givens_first"
                else rf"Find E\[T\] in minutes\. Given T is uniform on \[\((?P<lower>{num})\), \((?P<upper>{num})\)\] minutes\."
            )
            match = re.fullmatch(pattern, body)
            if not match:
                raise ValueError("Uniform-mean prompt does not match the frozen direct grammar")
            return {
                "kind": "uniform",
                "conditional": False,
                "lower": match["lower"],
                "cutoff": match["lower"],
                "upper": match["upper"],
            }
        pattern = (
            rf"T is uniform on \[\((?P<lower>{num})\), \((?P<upper>{num})\)\] minutes\. Given T>\((?P<cutoff>{num})\) (?P<unit>seconds|minutes), find the conditional mean of total T in minutes\."
            if surface_form == "givens_first"
            else rf"Find the conditional mean of total T in minutes, given T>\((?P<cutoff>{num})\) (?P<unit>seconds|minutes) and T uniform on \[\((?P<lower>{num})\), \((?P<upper>{num})\)\] minutes\."
        )
        match = re.fullmatch(pattern, body)
        if not match:
            raise ValueError("Conditional-uniform prompt does not match the frozen direct grammar")
        cutoff = F(match["cutoff"]) / 60 if match["unit"] == "seconds" else F(match["cutoff"])
        return {
            "kind": "uniform",
            "conditional": True,
            "lower": match["lower"],
            "cutoff": qstr(cutoff),
            "upper": match["upper"],
        }
    if category == "binomial":
        pattern = (
            rf"X is the number of successes in \((?P<n>{num})\) independent trials, each with success probability \((?P<p>{num})\)\. Find P\(X=\((?P<r>{num})\)\)\."
            if surface_form == "givens_first"
            else rf"Find P\(X=\((?P<r>{num})\)\)\. Given X counts successes in \((?P<n>{num})\) independent trials with success probability \((?P<p>{num})\) per trial\."
        )
        match = re.fullmatch(pattern, body)
        if not match:
            raise ValueError("Binomial prompt does not match the frozen direct grammar")
        return {"kind": "binomial", **match.groupdict()}
    if category == "interval":
        pattern = (
            rf"A normal-theory confidence interval is \[\((?P<lower>{num})\), \((?P<upper>{num})\)\]\. The sample size is multiplied by \((?P<multiplier>{num})\); the center, confidence level, and population standard deviation stay fixed\. Find the new upper endpoint\."
            if surface_form == "givens_first"
            else rf"Find the new upper endpoint, given a normal-theory confidence interval is \[\((?P<lower>{num})\), \((?P<upper>{num})\)\], the sample size is multiplied by \((?P<multiplier>{num})\), and the center, confidence level, and population standard deviation stay fixed\."
        )
        match = re.fullmatch(pattern, body)
        if not match:
            raise ValueError("Interval prompt does not match the frozen direct grammar")
        return {
            "kind": "interval",
            "lower": match["lower"],
            "upper": match["upper"],
            "divisor": qstr(rational_sqrt(match["multiplier"])),
        }
    raise ValueError(category)


def validate_prompt_semantics(row):
    s = row["semantics"]
    parsed = parse_question_semantics(
        row["question"], row["category"], row["coverage_axes"]["surface_form"]
    )
    if canonical_semantics(parsed) != canonical_semantics(s):
        raise ValueError("Independently parsed prompt facts disagree with semantics: " + row["id"])
    expected = build_question(row["category"], s, row["coverage_axes"])
    if row["question"] != expected:
        raise ValueError("Prompt text disagrees with semantics: " + row["id"])
    if row["prompt"] != row["question"] + SUFFIX:
        raise ValueError("Prompt suffix drift: " + row["id"])
    if row["template_signature"] != template_signature(row["question"], row["prompt"]):
        raise ValueError("Template signature disagrees with frozen normalizer: " + row["id"])
    if row["prompt_bindings"] != prompt_bindings(row["category"], s, row["coverage_axes"]):
        raise ValueError("Parsed prompt bindings disagree with semantics: " + row["id"])
    if row["category"] == "uniform_mean":
        if s.get("conditional") is not False or F(s["cutoff"]) != F(s["lower"]):
            raise ValueError("Unconditional uniform rows require cutoff == lower: " + row["id"])
        if "Given T>" in row["question"]:
            raise ValueError("Unconditional uniform wording contains a condition: " + row["id"])
    if row["category"] == "uniform_conditional":
        if s.get("conditional") is not True or "conditional mean of total T" not in row["question"]:
            raise ValueError("Conditional uniform flag/wording mismatch: " + row["id"])
    if row["category"] in PROCESS_CATEGORIES and row["coverage_axes"]["duration_unit"] == "seconds":
        shown = row["prompt_bindings"]["display_duration"]
        if F(shown) / 60 != F(s["duration"]):
            raise ValueError("Displayed seconds disagree with duration in minutes: " + row["id"])
    if row["category"] == "uniform_conditional" and row["coverage_axes"]["cutoff_unit"] == "seconds":
        shown = row["prompt_bindings"]["display_cutoff"]
        if F(shown) / 60 != F(s["cutoff"]):
            raise ValueError("Displayed cutoff seconds disagree with semantics: " + row["id"])
    if row["category"] == "interval":
        if F(row["prompt_bindings"]["sample_size_multiplier"]) != F(s["divisor"]) ** 2:
            raise ValueError("Displayed sample multiplier disagrees with width divisor: " + row["id"])
    return digest(
        {
            "question": row["question"],
            "parsed_semantics": canonical_semantics(parsed),
            "semantics": canonical_semantics(s),
            "bindings": row["prompt_bindings"],
        }
    )


def verify_row(row):
    prompt_hash = validate_prompt_semantics(row)
    expected = oracle(row["semantics"])
    if F(row["answer"]) != expected or F(calculate(row["expression"])) != expected:
        raise ValueError("Expression/answer disagrees with independent oracle: " + row["id"])
    if row["target"] != "Expression: " + row["expression"] or "\n" in row["target"]:
        raise ValueError("Target is not one line: " + row["id"])
    validate_question(row)
    if as_jsonable(task_key(row)) != row["semantic_key"]:
        raise ValueError("Stored semantic key drift: " + row["id"])
    judged = score(row["target"], row)
    if not judged["primary_correct"] or not judged["strict_one_line_expression"]:
        raise ValueError(("Reference target failed isolated grader", row["id"], judged))
    expected_traps = expected_misconception_audit(row)
    if row["misconception_traps"] != expected_traps:
        raise ValueError("Stored misconception mutation audit drift: " + row["id"])
    expected_discriminative = [
        {
            "id": trap["id"],
            "expression": trap["expression"],
            "computed": trap["computed"],
            "structure_differs_from_reference": trap["structure_differs_from_reference"],
            "grader_primary_correct": trap["grader_primary_correct"],
        }
        for trap in expected_traps
        if not trap["collides_with_oracle"]
        and trap["structure_differs_from_reference"]
        and not trap["grader_primary_correct"]
    ]
    if row["discriminative_mutations"] != expected_discriminative:
        raise ValueError("Stored discriminative mutations drift: " + row["id"])
    row["verification"] = {
        "prompt_semantics_consistent": True,
        "independent_prompt_parser_passed": True,
        "prompt_parser_version": PROMPT_PARSER_VERSION,
        "prompt_semantics_sha256": prompt_hash,
        "template_normalizer_version": TEMPLATE_NORMALIZER_VERSION,
        "template_signature_recomputed": True,
        "independent_oracle_correct": True,
        "oracle_value": str(expected),
        "calculator_matches_oracle": True,
        "grader_primary_correct": True,
        "strict_one_line_expression": True,
        "misconception_mutations_checked": len(expected_traps),
        "discriminative_mutations_checked": len(expected_discriminative),
        "semantics_version": SEMANTICS_VERSION,
        "grader_version": GRADER_VERSION,
    }


def build_family_rows(
    split,
    size,
    historical_tasks,
    historical_projections,
    historical_questions,
    historical_prompts,
    generated_tasks,
    generated_projections,
    generated_groups,
    generated_questions,
    generated_prompts,
):
    result = {family: {} for family in FAMILY_ORDER}

    def accepted(generator, family, index):
        for attempt in range(10000):
            try:
                rows, group_key = generator(split, index, attempt)
            except ValueError:
                continue
            for contrast_position, row in enumerate(rows):
                row["contrast_position"] = contrast_position
            if accept_candidate(
                rows,
                group_key,
                split,
                historical_tasks,
                historical_projections,
                historical_questions,
                historical_prompts,
                generated_tasks,
                generated_projections,
                generated_groups,
                generated_questions,
                generated_prompts,
            ):
                return rows
        raise RuntimeError(("Could not create collision-free group", split, family, index))

    for index in range(size):
        for family, generator in (
            ("events", event_candidate),
            ("moments", moment_candidate),
            ("poisson", poisson_candidate),
            ("process", process_candidate),
            ("uniform", uniform_candidate),
        ):
            result[family][index] = accepted(generator, family, index)

    for start in range(0, size, 5):
        indices = list(range(start, min(start + 5, size)))
        rows = accepted(
            lambda current_split, group_index, attempt: binomial_group_candidate(
                current_split, group_index, indices, attempt
            ),
            "binomial",
            start // 5,
        )
        for row in rows:
            result["binomial"][int(row["id"].rsplit("_", 1)[1])] = [row]

    interval_group_size = 4 if split == "train" else 2
    for start in range(0, size, interval_group_size):
        indices = list(range(start, min(start + interval_group_size, size)))
        rows = accepted(
            lambda current_split, group_index, attempt: interval_group_candidate(
                current_split, group_index, indices, attempt
            ),
            "interval",
            start // interval_group_size,
        )
        for row in rows:
            result["interval"][int(row["id"].rsplit("_", 1)[1])] = [row]
    return result


def ordered_split(family_rows, size, split):
    rows = []
    for cycle in range(size):
        queues = {}
        for family in FAMILY_ORDER:
            group = list(family_rows[family][cycle])
            if len(group) > 1:
                inner_shift = cycle % len(group)
                group = group[inner_shift:] + group[:inner_shift]
            queues[family] = iter(group)
        cycle_rows = []
        for family in CYCLE_FAMILY_SCHEDULE:
            cycle_rows.append(next(queues[family]))
        for family, queue in queues.items():
            try:
                next(queue)
            except StopIteration:
                continue
            raise RuntimeError(("Family schedule omitted a category", split, cycle, family))
        if len(cycle_rows) != len(CATEGORIES) or set(row["category"] for row in cycle_rows) != set(CATEGORIES):
            raise RuntimeError(("Curriculum cycle is not category-balanced", split, cycle))
        for position, row in enumerate(cycle_rows):
            row["curriculum_cycle"] = cycle
            row["order_within_cycle"] = position
            attach_misconception_audit(row)
            verify_row(row)
        rows.extend(cycle_rows)
    return rows


def axis_audit(rows):
    names = sorted({name for row in rows for name in row["coverage_axes"]})
    audit = {}
    for name in names:
        observations = [
            (row["id"], row["coverage_axes"][name])
            for row in rows
            if name in row["coverage_axes"]
        ]
        audit[name] = {
            "applicable_rows": len(observations),
            "counts": dict(sorted(Counter(value for _, value in observations).items())),
            "sha256": digest(observations),
        }
    return audit


def category_axis_audit(rows):
    result = {}
    for category in CATEGORIES:
        category_rows = [row for row in rows if row["category"] == category]
        result[category] = {
            "row_count": len(category_rows),
            "axes": axis_audit(category_rows),
            "sha256": digest(
                [(row["id"], row["coverage_axes"]) for row in category_rows]
            ),
        }
    return result


def trap_audit(rows):
    observations = []
    counts = Counter()
    collisions = Counter()
    for row in rows:
        for trap in row["misconception_traps"]:
            observations.append((row["id"], trap["id"], trap["audit_status"], trap["computed"]))
            counts[trap["id"]] += 1
            if trap["collides_with_oracle"]:
                collisions[trap["id"]] += 1
    return {
        "counts": dict(sorted(counts.items())),
        "degenerate_collision_counts": dict(sorted(collisions.items())),
        "sha256": digest(observations),
    }


def discriminative_audit(rows, require_each_mutation=False):
    possible = Counter()
    discriminative = Counter()
    observations = []
    for row in rows:
        for trap in row["misconception_traps"]:
            key = f"{row['category']}::{trap['id']}"
            possible[key] += 1
            if not trap["collides_with_oracle"] and trap["structure_differs_from_reference"]:
                if trap["grader_primary_correct"]:
                    raise RuntimeError(("Mutation incorrectly receives primary credit", row["id"], trap["id"]))
                discriminative[key] += 1
                observations.append((row["id"], trap["id"], trap["expression"], trap["computed"]))
    missing = sorted(key for key in possible if not discriminative[key])
    if require_each_mutation and missing:
        raise RuntimeError(("Evaluation split lacks a nondegenerate misconception discriminator", missing))
    return {
        "possible_counts": dict(sorted(possible.items())),
        "discriminative_counts": dict(sorted(discriminative.items())),
        "missing_discriminators": missing,
        "sha256": digest(observations),
    }


def text_registry_audit(data):
    result = {}
    for split, rows in data.items():
        questions = [row["question"] for row in rows]
        prompts = [row["prompt"] for row in rows]
        signatures = [row["template_signature"] for row in rows]
        signature_counts = Counter(signatures)
        surface_counts = Counter(row["coverage_axes"]["surface_form"] for row in rows)
        for category in CATEGORIES:
            forms = {
                row["coverage_axes"]["surface_form"]
                for row in rows
                if row["category"] == category
            }
            if forms != set(CONTROLLED_SURFACE_FORMS):
                raise RuntimeError(("Category lacks both controlled surface forms", split, category, forms))
        max_allowed = math.ceil(len(rows) / len(CATEGORIES) / len(CONTROLLED_SURFACE_FORMS))
        if max(signature_counts.values()) > max_allowed:
            raise RuntimeError(("A normalized template is over-concentrated", split, max(signature_counts.values()), max_allowed))
        if len(signature_counts) < len(CATEGORIES) * len(CONTROLLED_SURFACE_FORMS):
            raise RuntimeError(("Too few controlled templates", split, len(signature_counts)))
        result[split] = {
            "question_count": len(questions),
            "unique_question_count": len(set(questions)),
            "question_registry_sha256": digest(sorted(questions)),
            "prompt_count": len(prompts),
            "unique_prompt_count": len(set(prompts)),
            "prompt_registry_sha256": digest(sorted(prompts)),
            "template_signature_counts": dict(sorted(signature_counts.items())),
            "unique_template_signature_count": len(signature_counts),
            "max_template_signature_reuse": max(signature_counts.values()),
            "max_template_signature_reuse_contract": max_allowed,
            "surface_form_counts": dict(sorted(surface_counts.items())),
            "template_signature_registry_sha256": digest(sorted(signatures)),
        }
    overlaps = {}
    split_names = list(data)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            left_questions = {row["question"] for row in data[left]}
            right_questions = {row["question"] for row in data[right]}
            left_prompts = {row["prompt"] for row in data[left]}
            right_prompts = {row["prompt"] for row in data[right]}
            left_templates = {row["template_signature"] for row in data[left]}
            right_templates = {row["template_signature"] for row in data[right]}
            exact_questions = left_questions & right_questions
            exact_prompts = left_prompts & right_prompts
            if exact_questions or exact_prompts:
                raise RuntimeError(("Exact cross-split text collision", left, right))
            templates = sorted(left_templates & right_templates)
            overlaps[f"{left}__{right}"] = {
                "exact_question_count": 0,
                "exact_prompt_count": 0,
                "shared_template_signature_count": len(templates),
                "shared_template_signature_sha256": digest(templates),
            }
    result["cross_split_overlap"] = overlaps
    result["sha256"] = digest(result)
    return result


def ordering_audit(train):
    batches = [train[start : start + 8] for start in range(0, len(train), 8)]
    if any(len(batch) != 8 for batch in batches):
        raise RuntimeError("Training rows do not form complete 8-row microbatches")
    event_histogram = Counter()
    batch_observations = []
    for batch_index, batch in enumerate(batches):
        category_counts = Counter(row["category"] for row in batch)
        if max(category_counts.values()) > 2:
            raise RuntimeError(("A category exceeds two rows in a microbatch", batch_index))
        family_counts = Counter(CATEGORY_TO_FAMILY[row["category"]] for row in batch)
        event_histogram[family_counts["events"]] += 1
        if family_counts["events"] not in (2, 3):
            raise RuntimeError(("Event microbatch count outside frozen 2/3 contract", batch_index))
        if any(count > 2 for family, count in family_counts.items() if family != "events"):
            raise RuntimeError(("Non-event family exceeds two rows in a microbatch", batch_index))
        batch_observations.append((batch_index, dict(sorted(category_counts.items())), dict(sorted(family_counts.items()))))
    expected_three = len([row for row in train if row["category"] in EVENT_CATEGORIES]) - 2 * len(batches)
    expected_histogram = Counter({2: len(batches) - expected_three, 3: expected_three})
    if event_histogram != expected_histogram:
        raise RuntimeError(("Event microbatch histogram drift", event_histogram))
    windows = []
    for start in range(0, len(train), 288):
        window = train[start : start + 288]
        counts = Counter(row["category"] for row in window)
        expected_per_category = len(window) // len(CATEGORIES)
        if len(window) % len(CATEGORIES) or any(counts[category] != expected_per_category for category in CATEGORIES):
            raise RuntimeError(("A curriculum window is not exactly category-balanced", start, counts))
        windows.append((start, dict(sorted(counts.items()))))
    sliding_family_max = {
        family: max(
            sum(CATEGORY_TO_FAMILY[row["category"]] == family for row in train[start : start + 8])
            for start in range(0, len(train) - 7)
        )
        for family in FAMILY_ORDER
    }
    return {
        "microbatch_size": 8,
        "microbatch_count": len(batches),
        "event_rows_per_microbatch_histogram": {
            str(count): batches for count, batches in sorted(event_histogram.items())
        },
        "non_event_family_max_per_microbatch": 2,
        "category_max_per_microbatch": 2,
        "microbatch_registry_sha256": digest(batch_observations),
        "sliding_8_row_family_max": sliding_family_max,
        "window_size": 288,
        "window_count": len(windows),
        "full_window_each_category": 16,
        "tail_window_rows": len(train) % 288,
        "tail_window_each_category": (len(train) % 288) // len(CATEGORIES),
        "window_registry_sha256": digest(windows),
    }


def held_out_combination_audit(data):
    audit = {}
    for name, required in HELD_OUT_STRUCTURAL_COMBINATIONS.items():
        def matches(row):
            values = {"family": CATEGORY_TO_FAMILY[row["category"]], **row["coverage_axes"]}
            return all(values.get(key) == value for key, value in required.items())

        counts = {split: sum(matches(row) for row in rows) for split, rows in data.items()}
        if counts["train"] != 0 or counts["development"] != 0 or counts["final_blind"] == 0:
            raise RuntimeError(("Predeclared held-out structural combination contract failed", name, counts))
        audit[name] = {
            "axis_requirements": required,
            "counts": counts,
            "final_match_commitment_sha256": digest(
                [row["id"] for row in data["final_blind"] if matches(row)]
            ),
        }
    return audit


def category_contracts():
    contracts = {}
    for category in CATEGORIES:
        if category in EVENT_CATEGORIES:
            design = "Five requested events share each independent A/B story; every target is correct."
        elif category in MOMENT_CATEGORIES:
            design = "Mean, variance, and second moment share each affine-transform story."
        elif category in POISSON_CATEGORIES:
            design = "Variance, scaled variance, and second moment share each Poisson-count story."
        elif category in PROCESS_CATEGORIES:
            design = "Variance, scaled variance, and second moment share each process story and duration."
        elif category in UNIFORM_CATEGORIES:
            design = "Unconditional and conditional-total means share support; no elapsed-wait target is mislabeled."
        elif category == "binomial":
            design = "r=0, 1, interior, n-1, and n are correct boundary/interior queries on shared trial stories."
        else:
            design = "Several rational sample-size multipliers are correct contrasts on the same original interval."
        contracts[category] = {
            "positive_target_only": True,
            "contrast_design": design,
            "misconception_mutations_are_audit_only": True,
        }
    return contracts


def split_summary(rows):
    return {
        "rows": len(rows),
        "category_counts": dict(sorted(Counter(row["category"] for row in rows).items())),
        "family_counts": dict(sorted(Counter(row["family"] for row in rows).items())),
        "group_story_count": len({canonical_key(row["group_story_key"]) for row in rows}),
        "task_key_count": len({canonical_key(row["semantic_key"]) for row in rows}),
    }


def contrast_group_audit(rows, split):
    """Reject correct aggregate counts built from incomplete contrast stories."""
    groups = {}
    for row in rows:
        groups.setdefault(row["contrast_group"], []).append(row)
    expected_categories = {
        "events": set(EVENT_CATEGORIES),
        "moments": set(MOMENT_CATEGORIES),
        "poisson": set(POISSON_CATEGORIES),
        "process": set(PROCESS_CATEGORIES),
        "uniform": set(UNIFORM_CATEGORIES),
        "binomial": {"binomial"},
        "interval": {"interval"},
    }
    expected_sizes = {
        "events": 5,
        "moments": 3,
        "poisson": 3,
        "process": 3,
        "uniform": 2,
        "binomial": 5,
        "interval": 4 if split == "train" else 2,
    }
    observations = []
    for group_id, group in sorted(groups.items()):
        families = {row["family"] for row in group}
        if len(families) != 1:
            raise RuntimeError(("Contrast group mixes families", split, group_id, families))
        family = next(iter(families))
        expected_size = expected_sizes[family]
        if len(group) != expected_size:
            raise RuntimeError(("Incomplete contrast group", split, family, group_id, len(group), expected_size))
        if any(row["contrast_size"] != expected_size for row in group):
            raise RuntimeError(("Stored contrast size disagrees with group", split, group_id))
        if {row["contrast_position"] for row in group} != set(range(expected_size)):
            raise RuntimeError(("Contrast positions are incomplete", split, group_id))
        categories = {row["category"] for row in group}
        if family != "binomial" and family != "interval" and categories != expected_categories[family]:
            raise RuntimeError(("Contrast categories are incomplete", split, family, group_id, categories))
        if family == "binomial":
            slots = {row["coverage_axes"]["r_slot"] for row in group}
            required = {"r_zero", "r_one", "r_interior", "r_n_minus_one", "r_n"}
            if slots != required:
                raise RuntimeError(("Binomial r slots are incomplete", split, group_id, slots))
            if len({(row["semantics"]["n"], row["semantics"]["p"]) for row in group}) != 1:
                raise RuntimeError(("Binomial story changes n or p", split, group_id))
        observations.append((group_id, family, expected_size, sorted(categories)))
    return {
        "group_count": len(groups),
        "family_group_counts": dict(sorted(Counter(item[1] for item in observations).items())),
        "all_groups_complete": True,
        "sha256": digest(observations),
    }


def build(output_path=DEFAULT_OUTPUT, blind_output_path=DEFAULT_BLIND_OUTPUT):
    (
        historical_tasks,
        historical_projections,
        historical_questions,
        historical_prompts,
        historical_audit,
    ) = historical_registry((output_path, blind_output_path))
    generated_tasks = set()
    generated_projections = {}
    generated_groups = {}
    generated_questions = set()
    generated_prompts = set()
    data = {}
    for split, size in SPLIT_SIZES.items():
        family_rows = build_family_rows(
            split,
            size,
            historical_tasks,
            historical_projections,
            historical_questions,
            historical_prompts,
            generated_tasks,
            generated_projections,
            generated_groups,
            generated_questions,
            generated_prompts,
        )
        data[split] = ordered_split(family_rows, size, split)

    # The stronger primitive key catches, for example, a Poisson mean reused in
    # another split even if one row additionally defines a scale and offset.
    task_owners, story_owners = {}, {}
    for split in SPLIT_SIZES:
        for row in data[split]:
            task_string = canonical_key(row["semantic_key"])
            if task_string in task_owners:
                raise RuntimeError(("Duplicate generated task", task_owners[task_string], row["id"]))
            task_owners[task_string] = row["id"]
            story_string = canonical_key(row["group_story_key"])
            prior = story_owners.setdefault(story_string, split)
            if prior != split:
                raise RuntimeError(("Cross-split group story", prior, split, row["id"]))
        assert not {canonical_key(row["semantic_key"]) for row in data[split]} & historical_tasks

    split_rows = {split: data[split] for split in SPLIT_SIZES}
    split_sha256 = {split: digest(split_rows[split]) for split in SPLIT_SIZES}
    order_sha256 = {split: digest([row["id"] for row in split_rows[split]]) for split in SPLIT_SIZES}
    semantic_key_sha256 = {
        split: digest([row["semantic_key"] for row in split_rows[split]]) for split in SPLIT_SIZES
    }
    story_key_sha256 = {
        split: digest([row["group_story_key"] for row in split_rows[split]]) for split in SPLIT_SIZES
    }
    coverage = {split: axis_audit(split_rows[split]) for split in SPLIT_SIZES}
    category_coverage = {split: category_axis_audit(split_rows[split]) for split in SPLIT_SIZES}
    traps = {split: trap_audit(split_rows[split]) for split in SPLIT_SIZES}
    discriminative = {
        split: discriminative_audit(split_rows[split], require_each_mutation=split != "train")
        for split in SPLIT_SIZES
    }
    text_registry = text_registry_audit(split_rows)
    contrast_groups = {split: contrast_group_audit(rows, split) for split, rows in split_rows.items()}
    held_out = held_out_combination_audit(split_rows)
    split_streams = {
        split: {
            "domain_label_sha256": text_digest(f"{SEED}|{split}"),
            "derivation": "sha256(SEED|split|family|index|attempt) first 8 bytes -> Random seed",
            "shared_question_renderer": "build_question/givens_first_or_target_first",
        }
        for split in SPLIT_SIZES
    }
    train = data["train"]
    cycle_hashes = [digest(train[start : start + len(CATEGORIES)]) for start in range(0, len(train), len(CATEGORIES))]
    order_contract = ordering_audit(train)
    generation_streams = {
        split: {
            "derivation": "sha256(SEED|split|family|index|attempt) first 8 bytes as deterministic RNG seed",
            "split_domain_label_sha256": text_digest(f"{SEED}|{split}"),
            "family_domain_label_sha256": digest(
                {family: text_digest(f"{SEED}|{split}|{family}") for family in FAMILY_ORDER}
            ),
        }
        for split in SPLIT_SIZES
    }
    if len({item["split_domain_label_sha256"] for item in generation_streams.values()}) != len(SPLIT_SIZES):
        raise RuntimeError("Split generation streams are not domain-separated")
    blind_receipt_body = {
        "data_revision": DATA_REVISION,
        "split": "final_blind",
        "count": len(data["final_blind"]),
        "category_counts": split_summary(data["final_blind"])["category_counts"],
        "split_sha256": split_sha256["final_blind"],
        "order_sha256": order_sha256["final_blind"],
        "semantic_task_key_sha256": semantic_key_sha256["final_blind"],
        "group_story_key_sha256": story_key_sha256["final_blind"],
        "sealed_before_training": True,
        "target_authority": "canonical_symbolic_oracle",
        "prompt_calls_before_unsealing": 0,
        "generation_stream_sha256": generation_streams["final_blind"]["split_domain_label_sha256"],
        "generation_stream_domain_label_sha256": split_streams["final_blind"]["domain_label_sha256"],
    }
    blind_receipt = {**blind_receipt_body, "receipt_sha256": digest(blind_receipt_body)}
    blind_payload = {"final_blind": data["final_blind"], "receipt": blind_receipt}
    blind_artifact_sha256 = text_digest(canonical_json_text(blind_payload))
    manifest = {
        "data_revision": DATA_REVISION,
        "status": "frozen_before_training_or_final_blind_inference",
        "seed": SEED,
        "split_generation_streams": split_streams,
        "split_generation_streams_sha256": digest(split_streams),
        "categories": list(CATEGORIES),
        "split_sizes_per_category": SPLIT_SIZES,
        "counts": {split: len(data[split]) for split in SPLIT_SIZES},
        "split_summary": {split: split_summary(split_rows[split]) for split in SPLIT_SIZES},
        "split_sha256": split_sha256,
        "order_sha256": order_sha256,
        "semantic_task_key_sha256": semantic_key_sha256,
        "group_story_key_sha256": story_key_sha256,
        "exact_text_and_template_registry": text_registry,
        "contrast_group_audit": contrast_groups,
        "contrast_group_audit_sha256": digest(contrast_groups),
        "coverage_axis_audit": coverage,
        "coverage_axis_audit_sha256": digest(coverage),
        "category_coverage_axis_audit": category_coverage,
        "category_coverage_axis_audit_sha256": digest(category_coverage),
        "misconception_trap_audit": traps,
        "misconception_trap_audit_sha256": digest(traps),
        "discriminative_mutation_audit": discriminative,
        "discriminative_mutation_audit_sha256": digest(discriminative),
        "predeclared_held_out_structural_combinations": held_out,
        "predeclared_held_out_structural_combinations_sha256": digest(held_out),
        "generation_streams": generation_streams,
        "generation_streams_sha256": digest(generation_streams),
        "non_selection_fuzz_contract": {
            "seed": 31415,
            "purpose": "discarded preflight portability and invariant check only",
            "eligible_for_training": False,
            "eligible_for_model_or_seed_selection": False,
            "artifact_persisted": False,
            "required_test": "DiverseCurriculumTests.test_discarded_31415_fuzz_build",
        },
        "category_contracts": category_contracts(),
        "ordering": {
            "train_strategy": "40 balanced 18-row cycles; same-story contrasts stay in one cycle and are interleaved across families",
            "cycle_size": len(CATEGORIES),
            "cycle_count": SPLIT_SIZES["train"],
            "every_cycle_has_each_category_once": True,
            "cycle_sha256": cycle_hashes,
            "cycle_sha256_digest": digest(cycle_hashes),
            "last_72_category_counts": dict(sorted(Counter(row["category"] for row in train[-72:]).items())),
            "last_144_category_counts": dict(sorted(Counter(row["category"] for row in train[-144:]).items())),
            "microbatch_and_window_audit": order_contract,
        },
        "historical_collision_registry": historical_audit,
        "final_blind_commitment": {
            "path": "docs/STATS_DIVERSE_FINAL_BLIND.json",
            "count": len(data["final_blind"]),
            "split_sha256": split_sha256["final_blind"],
            "receipt_sha256": blind_receipt["receipt_sha256"],
            "artifact_file_sha256": blind_artifact_sha256,
            "receipt": blind_receipt,
        },
        "isolation": {
            "training_source_splits": ["train"],
            "development_used_for_training": False,
            "final_blind_used_for_training": False,
            "teacher_allowed_input_splits": ["train"],
            "teacher_calls_during_dataset_build": 0,
            "teacher_payload_embedded": False,
            "development_teacher_eligible": False,
            "final_blind_teacher_eligible": False,
            "train_record_ids_sha256": digest([row["id"] for row in data["train"]]),
            "final_blind_rows_physically_separate": True,
            "public_artifact_top_level_keys": ["train", "development", "manifest"],
            "blind_artifact_top_level_keys": ["final_blind", "receipt"],
            "legacy_retention_required": True,
            "legacy_retention_artifact": "external_runner_input_not_produced_by_this_builder",
            "training_replay_required": True,
            "training_replay_artifact": "docs/STATS_DIVERSE_REPLAY.json",
            "training_replay_count": 720,
            "training_replay_source_split": "historical_train_only",
            "new_to_replay_ratio": "1:1",
            "external_24_probe_bundle": "external_existing_input_not_produced_by_this_builder",
        },
        "verification_contract": {
            "prompt_binding": "Independent frozen regex parser plus exact renderer and unit-aware semantic assertions; not validate_question alone.",
            "prompt_parser_version": PROMPT_PARSER_VERSION,
            "template_normalizer_version": TEMPLATE_NORMALIZER_VERSION,
            "row_oracle": "stats_consolidation_semantics.oracle",
            "row_grader": "stats_consolidation_grader.score primary_correct and strict one-line target",
            "semantics_version": SEMANTICS_VERSION,
            "grader_version": GRADER_VERSION,
            "grader_sha256": grader_fingerprint(),
            "builder_checker_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "importable_checker_apis": [
                "parse_question_semantics",
                "validate_prompt_semantics",
                "expected_misconception_audit",
                "verify_row",
                "normalize_template",
                "template_signature",
            ],
            "all_rows_verified": True,
        },
        "provenance": {
            "record_provenance": PROVENANCE,
            "target_authority": "canonical_symbolic_oracle",
            "authoring": "Assistant-authored parameter and coverage-slot expansion of independently verified statistical rules.",
            "teacher_role": "Optional train-only rule/surface review; never target authority and never final_blind input.",
            "new_cloud_teacher_calls_for_this_artifact": 0,
            "teacher_anchor_claims": False,
            "raw_teacher_errors_policy": "If a later train-only review disagrees, preserve raw output and record correction provenance; retain only oracle-verified targets.",
        },
    }
    data["manifest"] = manifest
    return data


def canonical_json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def public_artifact(data):
    return {"train": data["train"], "development": data["development"], "manifest": data["manifest"]}


def blind_artifact(data):
    receipt = data["manifest"]["final_blind_commitment"]["receipt"]
    payload = {"final_blind": data["final_blind"], "receipt": receipt}
    if text_digest(canonical_json_text(payload)) != data["manifest"]["final_blind_commitment"]["artifact_file_sha256"]:
        raise RuntimeError("Blind artifact commitment drift")
    return payload


def load_public_curriculum(path=DEFAULT_OUTPUT):
    """Preselection-safe loader: it never resolves or opens the blind path."""
    payload = json.loads(Path(path).read_text())
    if set(payload) != {"train", "development", "manifest"}:
        raise ValueError("Public curriculum must not contain final_blind bytes")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--blind-output", type=Path, default=DEFAULT_BLIND_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Build and compare with an existing frozen artifact")
    args = parser.parse_args()
    built = build(args.output, args.blind_output)
    public = public_artifact(built)
    blind = blind_artifact(built)
    if args.check:
        if json.loads(args.output.read_text()) != public:
            raise SystemExit("Frozen public curriculum differs from deterministic rebuild")
        if json.loads(args.blind_output.read_text()) != blind:
            raise SystemExit("Frozen blind curriculum differs from deterministic rebuild")
        if text_digest(args.blind_output.read_text()) != built["manifest"]["final_blind_commitment"]["artifact_file_sha256"]:
            raise SystemExit("Frozen blind file bytes differ from the public commitment")
    else:
        args.blind_output.write_text(canonical_json_text(blind))
        args.output.write_text(canonical_json_text(public))
    print(json.dumps(built["manifest"]["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
