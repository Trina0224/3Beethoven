import unittest

from evaluate_stats_consolidation import promotion


def suite(names, each):
    return {"n": len(names) * each, "correct": len(names) * each, "pending": 0,
            "category_counts": {name: {"n": each, "correct": each} for name in names}}


class FinalGateTests(unittest.TestCase):
    def test_fixed_gain_and_retention(self):
        old = suite([f"o{i}" for i in range(8)], 12)
        chain = suite([f"c{i}" for i in range(4)], 24)
        event = suite(["exactly_one", "both", "neither", "same"], 24)
        event["correct"] = 60; event["category_counts"]["exactly_one"]["correct"] = 12
        baseline = {"old": old, "chain": chain, "event": event, "mc": {"n": 240, "correct": 126}}
        candidate = {"old": suite([f"o{i}" for i in range(8)], 12),
                     "chain": suite([f"c{i}" for i in range(4)], 24),
                     "event": suite(["exactly_one", "both", "neither", "same"], 24),
                     "mc": {"n": 240, "correct": 122}}
        rule = {"old": {"n": 96, "allowed_total_loss": 4, "allowed_per_category_loss": 2},
                "chain": {"n": 96, "allowed_total_loss": 4, "allowed_per_category_loss": 2},
                "event": {"n": 96, "minimum_correct_total": 72, "minimum_gain_total": 8,
                          "exactly_one_n": 24, "exactly_one_minimum_correct": 18, "exactly_one_minimum_gain": 4},
                "mc": {"allowed_total_loss": 4}}
        self.assertTrue(promotion(candidate, baseline, rule)["passed"])
        candidate["mc"]["correct"] = 121
        self.assertFalse(promotion(candidate, baseline, rule)["passed"])


if __name__ == "__main__":
    unittest.main()
