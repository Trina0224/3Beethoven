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

GRADER_VERSION = "stats-consolidation-v2-reviewed-equivalences"


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
        return str(value.numerator) if value.denominator == 1 else f"({value.numerator}/{value.denominator})"
    if s["kind"] == "uniform":
        t, u = "("+s["cutoff"]+")", "("+s["upper"]+")"
        # elapsed + expected remaining == conditional TOTAL mean.
        refs += [f"{t}+({u}-{t})/2", f"{u}-({u}-{t})/2",
                 f"{t}/2+{u}/2", f"({u}**2-{t}**2)/(2*({u}-{t}))"]
    if s["kind"] == "interval":
        lo, hi, k = ("("+s[x]+")" for x in ("lower", "upper", "divisor"))
        refs += [f"{hi}-({hi}-{lo})/2+({hi}-{lo})/(2*{k})",
                 f"({lo}+{hi})/2+(({hi}-{lo})/2)/{k}"]
    if s["kind"] == "process" and s["target"] == "second_moment":
        rate, duration = "("+s["rate"]+")", "("+s["duration"]+")"
        refs += [f"{rate}*{duration}*(1+{rate}*{duration})"]
        if "/" in s["duration"]:
            numerator, denominator = s["duration"].split("/", 1)
            converted = f"({s['rate']}/{denominator})*{numerator}"
            refs += [f"{converted}*(1+{converted})"]
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
    judged["grader_version"] = GRADER_VERSION
    judged["raw_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    judged["strict_one_line_expression"] = bool(
        re.fullmatch(r"\s*Expression:\s*[^\n]+\s*", raw)
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
