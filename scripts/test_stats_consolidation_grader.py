import unittest

from stats_consolidation_pilot import build
from stats_curriculum_v0_19 import score


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


if __name__ == "__main__":
    unittest.main()
