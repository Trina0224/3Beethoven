"""Domain checks and an independent rational-number oracle for this experiment.

The oracle does not evaluate the reference expression. Semantic task keys use
only inputs needed by the requested quantity, not a whole story or its ID.
Historical source files are never modified by these adapters.
"""
import re
from fractions import Fraction as F

from exact_calculator import calculate

SEMANTICS_VERSION = "consolidation-domain-v2"
EVENTS = {"exactly_one", "at_least_one", "both", "neither", "same"}


def spec_for(q):
    if "semantics" in q:
        return dict(q["semantics"])
    b, c = q.get("bindings", {}), q["category"]
    if q.get("track"):
        a, minutes, scale, offset, variance, upper, cut = q["parameters"]
        t, d = q["track"], q["depth"]
        if t == "conditional_wait":
            return dict(kind="uniform", upper=str(upper), lower=str(cut if d == 1 else 0),
                        cutoff=str(cut) if d < 3 else f"{cut * 60 + 30}/60",
                        conditional=d != 1)
        target = "second_moment" if t == "second_moment" else "variance"
        extra = dict(scale=str(scale), offset=str(offset)) if t == "scaled_variance" else {}
        if (t == "scaled_variance" and d == 1) or (t == "second_moment" and d == 1):
            return dict(kind="moments", target=target, mean=str(a), variance=str(variance), **extra)
        process = d == 3 or (t == "poisson_variance" and d == 2)
        if process:
            duration = f"{minutes * 60 + 30}/60" if t == "poisson_variance" and d == 3 else str(minutes)
            return dict(kind="process", target=target, rate=str(a), duration=duration, **extra)
        return dict(kind="poisson", target=target, mean=str(a), **extra)
    if c in EVENTS:
        if "miss_a" in b:
            return dict(kind="events", target=c, p=str(1-F(b["miss_a"])), p_b=str(1-F(b["miss_b"])))
        if len(q.get("parameters", [])) == 3:
            a, z, den = q["parameters"]
            return dict(kind="events", target=c, p=f"{a}/{den}", p_b=f"{z}/{den}")
    if c == "binomial":
        return dict(kind=c, n=b["n"], r=b["r"], p=b["reject_probability"])
    if c == "uniform_time":
        return dict(kind="uniform", lower="0", upper=b["upper_minutes"], cutoff=b["cutoff_minutes"], conditional=True)
    if c == "interval":
        return dict(kind=c, lower=b["lower"], upper=b["upper"], divisor=b["width_divisor"])
    if c == "poisson_time":
        return dict(kind="process", target="variance", rate=b["rate_per_minute"], duration=b["duration_minutes"])
    if c == "poisson_scaled":
        return dict(kind="poisson", target="variance", mean=b["mean"], scale=b["scale"], offset=b["offset"])
    if c == "moment":
        return dict(kind="moments", target="second_moment", **b)
    raise ValueError("Missing semantic specification: " + q["id"])


def oracle(spec):
    k = spec["kind"]
    def f(key, default=None):
        return F(spec[key] if key in spec else default)
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    if k in ("events", "binomial"):
        p = f("p")
        require(0 <= p <= 1, "Probability outside [0,1]")
        if k == "binomial":
            n, r = f("n"), f("r")
            require(n.denominator == r.denominator == 1 and 0 <= r <= n <= 10000, "Invalid binomial counts")
            n, r = int(n), int(r)
            # Direct product, independent of the generated comb(...) expression.
            choose = F(1)
            for i in range(1, r + 1):
                choose *= F(n - i + 1, i)
            return choose * p**r * (1-p)**(n-r)
        z = f("p_b")
        require(0 <= z <= 1, "Probability outside [0,1]")
        predicate = {"exactly_one": lambda a,b: a != b, "at_least_one": lambda a,b: a or b,
                     "both": lambda a,b: a and b, "neither": lambda a,b: not a and not b,
                     "same": lambda a,b: a == b}[spec["target"]]
        return sum((p if a else 1-p)*(z if b else 1-z)
                   for a in (0,1) for b in (0,1) if predicate(a,b))
    if k == "uniform":
        lo, hi, cut = f("lower"), f("upper"), f("cutoff")
        require(0 <= lo < hi and lo <= cut < hi, "Invalid uniform support or impossible conditioning")
        # Integral of t divided by the length of the conditional support.
        return (hi**2-cut**2)/(2*(hi-cut))
    if k == "interval":
        lo, hi, divisor = f("lower"), f("upper"), f("divisor")
        require(lo < hi and divisor > 0, "Invalid interval or sample-size factor")
        return lo + (hi-lo)/2 + (hi-lo)/2/divisor
    if k == "process":
        rate, duration = f("rate"), f("duration")
        require(rate >= 0 and duration > 0, "Invalid Poisson rate or duration")
        mean = variance = rate*duration
    elif k == "poisson":
        mean = variance = f("mean")
        require(mean >= 0, "Negative Poisson mean")
    elif k == "moments":
        mean, variance = f("mean", 0), f("variance", 0)
        require(variance >= 0, "Negative variance")
    else:
        raise ValueError("Unknown semantic family: " + k)
    scale, offset = f("scale", 1), f("offset", 0)
    target = spec["target"]
    if target == "variance":
        return scale*scale*variance
    if target == "mean":
        return scale*mean+offset
    if target == "second_moment":
        # Expanded E[(aX+b)^2], distinct from the generator's Var(Y)+E[Y]^2.
        return scale*scale*(variance+mean*mean)+2*scale*offset*mean+offset*offset
    raise ValueError("Unknown requested quantity")


def validate_question(q):
    spec = spec_for(q)
    expected = oracle(spec)
    percentages = re.findall(r"(-?\d+(?:\.\d+)?)\s*(?:%|percent\b)", q["question"], flags=re.I)
    for percent in percentages:
        if not 0 <= F(percent) <= 100:
            raise ValueError("Invalid probability in prompt: " + q["id"])
    if percentages and spec["kind"] == "binomial":
        if [F(x)/100 for x in percentages] != [F(spec["p"])]:
            raise ValueError("Prompt probability disagrees with semantic inputs: " + q["id"])
    if percentages and spec["kind"] == "events":
        expected_probabilities = [F(spec["p"]), F(spec["p_b"])]
        if "miss" in q["question"].lower():
            expected_probabilities = [1-x for x in expected_probabilities]
        if sorted(F(x)/100 for x in percentages) != sorted(expected_probabilities):
            raise ValueError("Prompt probabilities disagree with semantic inputs: " + q["id"])
    if spec["kind"] in ("poisson", "process") and "poisson" not in q["question"].lower():
        raise ValueError("Missing Poisson assumption: " + q["id"])
    if F(q["answer"]) != expected or F(calculate(q["expression"])) != expected:
        raise ValueError("Reference disagrees with independent oracle: " + q["id"])
    return spec


def task_key(q):
    s = spec_for(q)
    oracle(s)  # Invalid questions cannot enter a split registry.
    k = s["kind"]
    target = s.get("target", k)
    if k == "events":
        return (k, target, *sorted((str(F(s["p"])), str(F(s["p_b"])))))
    if k == "binomial":
        names = ("n", "r", "p")
    elif k == "uniform":
        return (k, bool(s["conditional"]), str(F(s["lower"])), str(F(s["upper"])), str(F(s["cutoff"])))
    elif k == "interval":
        names = ("lower", "upper", "divisor")
    else:
        names = (("rate", "duration") if k == "process" else
                 ("variance",) if k == "moments" and target == "variance" else
                 ("mean", "variance") if k == "moments" and target == "second_moment" else ("mean",))
        names += ("scale",) if target == "variance" else ("scale", "offset")
    defaults = {"scale": "1", "offset": "0"}
    values = tuple((name, str(F(s.get(name, defaults.get(name, "0"))))) for name in names)
    return (k, target, *values)


def assert_disjoint(splits):
    owners = {}
    for split, questions in splits.items():
        for q in questions:
            key = task_key(q)
            if key in owners and owners[key][0] != split:
                raise ValueError(f"Subtask leakage: {owners[key][1]} -> {q['id']}")
            owners.setdefault(key, (split, q["id"]))
    return len(owners)
