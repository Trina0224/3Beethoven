import unittest

from stats_consolidation_holdout import build
from stats_consolidation_pilot import build as build_candidate
from stats_consolidation_semantics import task_key, validate_question


class HoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.holdout, cls.benchmark = build()

    def test_counts_and_teacher_subset(self):
        self.assertEqual({k: len(v) for k, v in self.holdout["suites"].items()},
                         {"old": 96, "chain": 96, "event": 96})
        self.assertEqual(len(self.benchmark), 24)

    def test_no_candidate_expression_or_id_leakage(self):
        candidate = build_candidate()[0]
        candidate_expr = {q["expression"] for s in candidate for q in s["questions"]}
        candidate_tasks = {task_key(q) for s in candidate for q in s["questions"]}
        candidate_ids = {q["id"] for s in candidate for q in s["questions"]}
        held = [q for rows in self.holdout["suites"].values() for q in rows]
        self.assertTrue(candidate_expr.isdisjoint(q["expression"] for q in held))
        self.assertTrue(candidate_tasks.isdisjoint(task_key(q) for q in held))
        self.assertTrue(candidate_ids.isdisjoint(q["id"] for q in held))

    def test_event_has_24_distinct_parameter_stories(self):
        event = self.holdout["suites"]["event"]
        self.assertEqual(len({q["story_id"] for q in event}), 24)
        self.assertTrue(all(sum(x["story_id"] == q["story_id"] for x in event) == 4 for q in event))

    def test_all_holdout_and_benchmark_questions_satisfy_independent_oracle(self):
        for q in [q for rows in self.holdout["suites"].values() for q in rows] + self.benchmark:
            validate_question(q)


if __name__ == "__main__":
    unittest.main()
