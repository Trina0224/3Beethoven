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
        parameter_owners = {}
        for story in self.stories:
            self.assertEqual(owners.setdefault(story["lineage_id"], story["split"]), story["split"])
            key = (story["archetype"], story["parameter_signature"])
            self.assertEqual(parameter_owners.setdefault(key, story["split"]), story["split"])
        markers = {"conditions_first": "Conditions first:", "quantity_first": "Given:",
                   "operational_story": "An analyst records this operating condition:",
                   "validation_compact": "Write one executable expression",
                   "validation_reverse": "Requested result:"}
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
                         {"old": 48, "chain": 48, "event": 16})
        self.assertEqual(len({q["story_id"] for q in matrix["suites"]["event"]}), 4)
        for rows in matrix["suites"].values():
            self.assertTrue(all("_test_" not in q["id"] for q in rows))

    def test_reference_is_executable_and_structurally_credited(self):
        for story in self.stories:
            for q in story["questions"]:
                self.assertEqual(calculate(q["expression"]), q["answer"])
                judged = grade("Expression: " + q["expression"], q)
                self.assertTrue(judged["executable"], q["id"])
                self.assertIs(judged["math_correct"], True, q["id"])

    def test_every_question_is_self_contained_and_keeps_original_target(self):
        forbidden = ("same counter", "same wait", "same y=", "that display")
        for story in self.stories:
            for q in story["questions"]:
                lowered = q["question"].lower()
                self.assertFalse(any(fragment in lowered for fragment in forbidden), q["id"])
                if story["archetype"] == "poisson_process":
                    self.assertIn("poisson", lowered, q["id"])
                if q["category"] == "v18_conditional_wait":
                    self.assertIn("total wait", lowered, q["id"])
                    self.assertNotIn("additional wait", lowered, q["id"])

    def test_train_and_validation_surface_families_are_disjoint(self):
        train = {s["surface_style"] for s in self.stories if s["split"] == "train"}
        validation = {s["surface_style"] for s in self.stories if s["split"] == "validation"}
        self.assertGreaterEqual(len(train), 2)
        self.assertGreaterEqual(len(validation), 2)
        self.assertTrue(train.isdisjoint(validation))

    def test_event_reference_by_outcome_enumeration(self):
        event_stories = [s for s in self.stories if s["archetype"] == "detection_events"]
        for story in event_stories:
            questions = {q["category"]: q for q in story["questions"]}
            first = questions["both"]["question"].split("probabilities ", 1)[1].split(".", 1)[0]
            left, right = first.split(" and ")
            p = F(left.rstrip("%")) / 100 if "%" in left else F(left)
            q = F(right.rstrip("%")) / 100 if "%" in right else F(right)
            probs = {(a, b): (p if a else 1-p) * (q if b else 1-q) for a in (0, 1) for b in (0, 1)}
            predicates = {"exactly_one": lambda a,b: a != b, "both": lambda a,b: a and b,
                          "neither": lambda a,b: not a and not b, "same": lambda a,b: a == b}
            for category, predicate in predicates.items():
                expected = sum(v for key, v in probs.items() if predicate(*key))
                self.assertEqual(F(calculate(questions[category]["expression"])), expected)


if __name__ == "__main__":
    unittest.main()
