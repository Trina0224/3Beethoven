import unittest

from stats_consolidation_pilot import build
from stats_consolidation_grader import score


class ConsolidationGraderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stories, _, _, _ = build()
        cls.questions = [q for story in stories for q in story["questions"]]

    def one(self, category):
        return next(q for q in self.questions if q["category"] == category)

    def test_reassociated_unit_conversion_is_credited(self):
        q = self.one("poisson_time")
        rate = q["bindings"]["rate_per_minute"]
        seconds = q["bindings"]["duration_minutes"].split("/", 1)[0]
        result = score(f"Expression: {rate}*{seconds}/60", q)
        self.assertTrue(result["executable"])
        self.assertTrue(result["math_correct"])

    def test_missing_unit_conversion_is_wrong(self):
        q = self.one("poisson_time")
        rate = q["bindings"]["rate_per_minute"]
        seconds = q["bindings"]["duration_minutes"].split("/", 1)[0]
        result = score(f"Expression: {rate}*{seconds}", q)
        self.assertTrue(result["executable"])
        self.assertFalse(result["math_correct"])

    def test_missing_variance_term_is_wrong(self):
        q = self.one("moment")
        b = q["bindings"]
        result = score(f"Expression: ({b['scale']}*{b['mean']}+{b['offset']})**2", q)
        self.assertTrue(result["executable"])
        self.assertFalse(result["math_correct"])

    def test_event_confusion_is_wrong(self):
        q = self.one("exactly_one")
        story = next(s for s in build()[0] if s["story_id"] == q["story_id"])
        same = next(x for x in story["questions"] if x["category"] == "same")
        result = score("Expression: " + same["expression"], q)
        self.assertTrue(result["executable"])
        self.assertFalse(result["math_correct"])

    def test_unsafe_expression_stays_uncredited(self):
        result = score('Expression: __import__("os").system("false")', self.one("moment"))
        self.assertFalse(result["executable"])
        self.assertIsNone(result["math_correct"])

    def test_primary_credit_and_format_are_separate(self):
        q = self.one("moment")
        result = score("Some prose\nExpression: " + q["expression"], q)
        self.assertTrue(result["primary_correct"])
        self.assertFalse(result["strict_one_line_expression"])

    def test_reviewed_factored_second_moment_gets_credit(self):
        q = next(q for q in self.questions if q["category"] == "v18_second_moment"
                 and q["semantics"]["kind"] == "process")
        s = q["semantics"]
        result = score(f"Expression: ({s['rate']})*({s['duration']})*(1+({s['rate']})*({s['duration']}))", q)
        self.assertTrue(result["primary_correct"])
        if "/" in s["duration"]:
            numerator, denominator = s["duration"].split("/", 1)
            converted = f"({s['rate']}/{denominator})*{numerator}"
            self.assertTrue(score(f"Expression: {converted}*(1+{converted})", q)["primary_correct"])

    def test_reviewed_folded_affine_coefficients_get_credit(self):
        q = self.one("poisson_scaled")
        b = q["bindings"]
        result = score(f"Expression: {int(b['scale'])**2}*{b['mean']}", q)
        self.assertTrue(result["primary_correct"])


if __name__ == "__main__":
    unittest.main()
