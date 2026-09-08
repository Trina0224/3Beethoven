"""Isolated grader contract for the consolidation experiment.

Historical graders remain frozen.  This wrapper records the exact raw response,
keeps mathematical/executable/format judgements separate, and defines the only
primary-credit rule used by the new selection and holdout evaluators.
"""
import ast
import hashlib
import re
from fractions import Fraction as F
from pathlib import Path

from stats_curriculum_v0_19 import score as historical_score
from formulation_grader import parse_expression, shape
from stats_consolidation_semantics import spec_for, validate_question
from exact_calculator import calculate

GRADER_VERSION = "stats-consolidation-v5-whole-raw-diagnostic"


_RAW_EXPRESSION_CHARACTERS = re.compile(r"(?:[0-9(),+\-*/\s]|comb)+\Z")


def validate_raw_expression(raw):
    """Return the unchanged executable expression and its exact value.

    The historical grader intentionally extracts a useful expression from
    labels, assignments, semicolon-delimited work, comments, and a few
    alternate syntaxes. That tolerance is useful for retrospective review,
    but it must not make malformed text an executable training target or a
    primary-success output. This contract therefore accepts only a complete
    one-line arithmetic expression, with an optional ``Expression:`` label.

    No rewriting is performed: ``^``, assignments, comments, prose, decimal
    literals, names, and functions other than ``comb`` are rejected. The
    bounded exact calculator provides the AST allowlist, arity checks, and
    resource limits.
    """
    if not isinstance(raw, str):
        raise TypeError("raw model response must be a string")
    text = raw.strip()
    if not text or "\n" in text or "\r" in text:
        raise ValueError("raw response must contain one non-empty line")
    labelled = re.fullmatch(r"Expression\s*:\s*(.+)", text, flags=re.IGNORECASE)
    expression = labelled.group(1).strip() if labelled else text
    if not expression or not _RAW_EXPRESSION_CHARACTERS.fullmatch(expression):
        raise ValueError("raw response is not an unchanged allowed expression")
    return expression, calculate(expression)


def grader_fingerprint():
    root = Path(__file__).parent
    names = ("stats_consolidation_grader.py", "stats_consolidation_semantics.py", "stats_teacher_envelope.py",
             "stats_curriculum_v0_19.py", "stats_curriculum_v0_18.py",
             "stats_curriculum_v0_13.py", "formulation_grader.py", "exact_calculator.py")
    return hashlib.sha256(b"\0".join(name.encode()+b"\0"+(root/name).read_bytes() for name in names)).hexdigest()


def reviewed_shape(expression):
    class Identities(ast.NodeTransformer):
        def visit_BinOp(self, node):
            node = self.generic_visit(node)
            if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant):
                if node.right.value == 1:
                    return node.left
                if node.right.value == 0:
                    return ast.Constant(value=1)
            if isinstance(node.op, ast.Mult):
                if isinstance(node.left, ast.Constant) and node.left.value == 1:
                    return node.right
                if isinstance(node.right, ast.Constant) and node.right.value == 1:
                    return node.left
            if isinstance(node.op, ast.Add):
                if isinstance(node.left, ast.Constant) and node.left.value == 0:
                    return node.right
                if isinstance(node.right, ast.Constant) and node.right.value == 0:
                    return node.left
            if (isinstance(node.op, ast.Sub) and isinstance(node.right, ast.Constant)
                    and node.right.value == 0):
                return node.left
            return node

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if (isinstance(node.func, ast.Name) and node.func.id == "comb"
                    and len(node.args) == 2 and all(isinstance(x, ast.Constant) for x in node.args)):
                n, r = [x.value for x in node.args]
                if type(n) is int and type(r) is int and 0 <= r <= n:
                    if r in (1, n-1):
                        return ast.Constant(value=n)
                    if r in (0, n):
                        return ast.Constant(value=1)
            return node
    return shape(Identities().visit(parse_expression(expression, {})))


def equivalent_references(q):
    s = spec_for(q)
    refs = [q["expression"]]
    def number(value):
        value = F(value)
        if value.denominator == 1:
            rendered = str(value.numerator)
            return f"({rendered})" if value < 0 else rendered
        return f"({value.numerator}/{value.denominator})"
    if s["kind"] == "events":
        p, z = "("+s["p"]+")", "("+s["p_b"]+")"
        target = s["target"]
        # Standard Boolean-algebra expansions.  These remain structural
        # references containing both supplied probabilities; numerical
        # agreement alone is still insufficient for credit.
        if target == "neither":
            refs += [f"1-{p}-{z}+{p}*{z}", f"1-({p}+{z}-{p}*{z})"]
        elif target == "at_least_one":
            refs += [f"{p}+{z}-{p}*{z}", f"{p}+(1-{p})*{z}",
                     f"{z}+(1-{z})*{p}"]
        elif target == "exactly_one":
            refs += [f"{p}+{z}-2*{p}*{z}",
                     f"({p}+{z}-{p}*{z})-{p}*{z}"]
        elif target == "same":
            refs += [f"1-{p}-{z}+2*{p}*{z}",
                     f"1-({p}+{z}-2*{p}*{z})"]
    if s["kind"] == "uniform":
        t, u = "("+s["cutoff"]+")", "("+s["upper"]+")"
        # elapsed + expected remaining == conditional TOTAL mean.
        refs += [f"{t}+({u}-{t})/2", f"{u}-({u}-{t})/2",
                 f"{t}/2+{u}/2", f"({u}**2-{t}**2)/(2*({u}-{t}))"]
    if s["kind"] == "interval":
        lo, hi, k = ("("+s[x]+")" for x in ("lower", "upper", "divisor"))
        refs += [f"{hi}-({hi}-{lo})/2+({hi}-{lo})/(2*{k})",
                 f"({lo}+{hi})/2+(({hi}-{lo})/2)/{k}"]
        center = number((F(s['lower'])+F(s['upper']))/2)
        half = number((F(s['upper'])-F(s['lower']))/2)
        refs += [f"{center}+{half}/{k}", f"{center}+({hi}-{lo})/(2*{k})",
                 f"({lo}+{hi})/2+{half}/{k}", f"{center}+({hi}-{center})/{k}"]
    if s["kind"] == "binomial":
        n, r, p = int(F(s["n"])), int(F(s["r"])), "("+s["p"]+")"
        # Boundary factors that are identically one may be omitted. These are
        # symbolic identities, not answer-only credit; wrong exponents or
        # coefficients still have different reviewed shapes and values.
        if r == 0:
            refs += [f"(1-{p})**{n}", f"{p}**0*(1-{p})**{n}",
                     f"comb({n},0)*(1-{p})**{n}"]
        elif r == n:
            refs += [f"{p}**{n}", f"comb({n},{n})*{p}**{n}",
                     f"{p}**{n}*(1-{p})**0"]
        elif r == 1:
            refs += [f"{n}*{p}*(1-{p})**({n}-1)",
                     f"{n}*{p}*(1-{p})**{n-1}"]
        elif r == n-1:
            refs += [f"{n}*{p}**({n}-1)*(1-{p})",
                     f"{n}*{p}**{n-1}*(1-{p})"]
    if s["kind"] == "process" and s["target"] == "second_moment":
        rate, duration = "("+s["rate"]+")", "("+s["duration"]+")"
        refs += [f"{rate}*{duration}*(1+{rate}*{duration})"]
        if "/" in s["duration"]:
            numerator, denominator = s["duration"].split("/", 1)
            converted = f"({s['rate']}/{denominator})*{numerator}"
            refs += [f"{converted}*(1+{converted})"]
    if s["kind"] == "poisson" and s["target"] == "second_moment":
        mean = number(F(s["mean"]))
        refs += [f"{mean}*({mean}+1)", f"{mean}+{mean}**2",
                 f"{mean}**2+{mean}"]
    if s["kind"] in ("poisson", "moments") and "scale" in s:
        scale = F(s["scale"])
        mean = F(s.get("mean", 0))
        offset = F(s.get("offset", 0))
        if s["target"] == "variance":
            variance = F(s.get("variance", s.get("mean", 0)))
            refs += [f"{number(scale*scale)}*{number(variance)}",
                     f"({number(scale)}*{number(scale)})*{number(variance)}"]
        if s["target"] == "second_moment":
            variance = F(s.get("variance", s.get("mean", 0)))
            refs += [
                f"{number(scale*scale)}*({number(variance)}+{number(mean)}**2)"
                f"+{number(2*scale*offset)}*{number(mean)}+{number(offset*offset)}",
                f"({number(scale)}**2)*({number(variance)}+{number(mean)}**2)"
                f"+2*{number(scale)}*{number(offset)}*{number(mean)}+{number(offset)}**2",
            ]
            refs += [f"{number(scale*scale)}*{number(variance)}+({number(scale)}*{number(mean)}+{number(offset)})**2"]
            # Fully distribute a^2 over Var(X)+E[X]^2.  This is a common,
            # exact presentation and was previously left as review-pending.
            refs += [
                f"{number(scale)}**2*{number(variance)}"
                f"+{number(scale)}**2*{number(mean)}**2"
                f"+2*{number(scale)}*{number(offset)}*{number(mean)}"
                f"+{number(offset)}**2",
                f"{number(scale*scale)}*{number(variance)}"
                f"+{number(scale*scale)}*{number(mean)}**2"
                f"+{number(2*scale*offset)}*{number(mean)}"
                f"+{number(offset*offset)}",
            ]
            for cross in (f"2*{number(scale)}*{number(offset)}*{number(mean)}",
                          f"{number(2*scale)}*{number(offset)}*{number(mean)}",
                          f"{number(2*scale*offset)}*{number(mean)}"):
                for square in (f"{number(offset)}**2", number(offset*offset)):
                    refs += [f"{number(scale*scale)}*({number(variance)}+{number(mean)}**2)+{cross}+{square}"]
    return refs


def score(raw, question):
    if not isinstance(raw, str):
        raise TypeError("raw model response must be a string")
    # Invalid problems must fail before scoring, even for a reference echo.
    validate_question(question)
    judged = dict(historical_score(raw, question))
    if (judged.get("math_correct") is None and judged.get("executable")
            and F(judged["computed"]) == F(question["answer"])):
        candidate = reviewed_shape(judged["normalized_expression"])
        if any(candidate == reviewed_shape(ref) for ref in equivalent_references(question)):
            judged.update(math_correct=True, review_required=False,
                          reason="Reviewed algebraic identity plus exact agreement; no answer-only credit.")
    historical_executable = judged.get("executable") is True
    try:
        raw_expression, raw_computed = validate_raw_expression(raw)
        raw_contract_error = None
        raw_executable = True
    except (TypeError, ValueError, SyntaxError, RecursionError, OverflowError) as exc:
        raw_expression = None
        raw_computed = None
        raw_contract_error = str(exc)
        raw_executable = False
    judged["grader_version"] = GRADER_VERSION
    judged["raw_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    judged["historical_executable_after_extraction"] = historical_executable
    judged["whole_raw_expression"] = raw_expression
    judged["whole_raw_computed"] = raw_computed
    judged["whole_raw_expression_contract_error"] = raw_contract_error
    judged["whole_raw_executable"] = raw_executable
    judged["strict_one_line_expression"] = bool(
        raw_executable and re.fullmatch(r"\s*Expression:\s*[^\n]+\s*", raw)
    )
    judged["primary_correct"] = bool(
        judged.get("math_correct") is True and judged.get("executable") is True
    )
    judged["correct"] = judged["primary_correct"]
    return judged


def outcome(judged):
    if judged["primary_correct"]:
        return "accepted"
    if not judged["executable"]:
        return "non_executable_or_format"
    if judged["math_correct"] is False:
        return "mathematical_error"
    return "equivalence_pending"
