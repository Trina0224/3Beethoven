import unittest

from stats_diverse_training_contract import (
    format_vector,
    load_training_plan,
    promotion_credit,
    promotion_vector,
)


class DiverseTrainingContractTests(unittest.TestCase):
    def test_actual_replay_is_half_of_every_training_pair(self):
        ordered, receipt = load_training_plan()
        self.assertEqual(len(ordered), 1440)
        self.assertEqual(receipt["new_count"], receipt["replay_count"])
        self.assertEqual(receipt["new_to_replay_ratio"], "1:1")
        for index in range(0, len(ordered), 2):
            self.assertEqual(ordered[index]["training_role"], "new_distillation")
            self.assertEqual(ordered[index + 1]["training_role"], "legacy_replay")
            self.assertEqual(ordered[index + 1]["row"]["source_split"], "train")

    def test_math_credit_cannot_be_replaced_by_strict_format(self):
        correct_with_prose = {
            "math_correct": True,
            "executable": True,
            "whole_raw_executable": False,
            "strict_one_line_expression": False,
        }
        wrong_but_strict = {
            "math_correct": False,
            "executable": True,
            "whole_raw_executable": True,
            "strict_one_line_expression": True,
        }
        self.assertTrue(promotion_credit(correct_with_prose))
        self.assertFalse(promotion_credit(wrong_but_strict))
        rows = [{"grader": correct_with_prose}, {"grader": wrong_but_strict}]
        self.assertEqual(promotion_vector(rows), [True, False])
        self.assertEqual(
            format_vector(rows),
            [
                {"whole_raw_executable": False, "strict_one_line_expression": False},
                {"whole_raw_executable": True, "strict_one_line_expression": True},
            ],
        )


if __name__ == "__main__":
    unittest.main()
