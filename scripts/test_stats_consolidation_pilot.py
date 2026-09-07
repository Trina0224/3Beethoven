import unittest
from collections import Counter
from fractions import Fraction as F

from exact_calculator import calculate
from formulation_grader import grade
from stats_consolidation_pilot import build, selection_validation


class ConsolidationPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stories, cls.requests, cls.coverage, cls.holdout = build()

    def test_counts_and_split_before_generation(self):
        self.assertEqual(len(self.stories), 96)
        self.assertEqual(sum(r["pilot_batch"] for r in self.requests), 24)
        self.assertEqual(Counter(s["split"] for s in self.stories), {"train": 71, "validation": 25})
        owners = {}
        for story in self.stories:
            self.assertEqual(owners.setdefault(story["lineage_id"], story["split"]), story["split"])
        markers = {"conditions_first": "Conditions first:", "quantity_first": "Given:",
                   "operational_story": "During an operations review,",
                   "symbolic_story": "Write a symbolic numerical setup",
                   "unit_emphasis": "Keep every stated unit explicit"}
        for style, marker in markers.items():
            self.assertTrue(all(marker in q["question"] for s in self.stories if s["surface_style"] == style for q in s["questions"]))

    def test_teacher_requests_contain_no_reference(self):
        forbidden = {"answer", "expression", "bindings", "reference"}
        for request in self.requests:
            for q in request["questions"]:
                self.assertFalse(forbidden & q.keys())
            self.assertNotIn("concrete_questions", self.holdout)

    def test_compact_selection_matrix_uses_validation_only(self):
        matrix = selection_validation(self.stories)
        self.assertEqual({k: len(v) for k, v in matrix["suites"].items()},
                         {"old": 32, "chain": 24, "event": 16, "candidate": 19})
        for rows in matrix["suites"].values():
            self.assertTrue(all("_test_" not in q["id"] for q in rows))

    def test_reference_is_executable_and_structurally_credited(self):
        for story in self.stories:
            for q in story["questions"]:
                self.assertEqual(calculate(q["expression"]), q["answer"])
                judged = grade("Expression: " + q["expression"], q)
                self.assertTrue(judged["executable"], q["id"])
                self.assertIs(judged["math_correct"], True, q["id"])

    def test_event_reference_by_outcome_enumeration(self):
        event_stories = [s for s in self.stories if s["archetype"] == "detection_events"]
        for story in event_stories:
            questions = {q["category"]: q for q in story["questions"]}
            first = questions["both"]["question"].split("probabilities ", 1)[1].split(".", 1)[0]
            left, right = first.split(" and ")
            p, q = F(left), F(right)
            probs = {(a, b): (p if a else 1-p) * (q if b else 1-q) for a in (0, 1) for b in (0, 1)}
            predicates = {"exactly_one": lambda a,b: a != b, "both": lambda a,b: a and b,
                          "neither": lambda a,b: not a and not b, "same": lambda a,b: a == b}
            for category, predicate in predicates.items():
                expected = sum(v for key, v in probs.items() if predicate(*key))
                self.assertEqual(F(calculate(questions[category]["expression"])), expected)


if __name__ == "__main__":
    unittest.main()
