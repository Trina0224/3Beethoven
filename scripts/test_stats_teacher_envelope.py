import json
import unittest
from stats_teacher_envelope import parse_answers


class EnvelopeTests(unittest.TestCase):
    def test_valid_and_repairs_preserve_expression(self):
        item = {"question_id": "q", "expression": "3**2*(79**2+79)+2*3*5*79+5**2"}
        for payload, suffix in [({"answers": [item]}, ""), ({"answers": [item]}, "}"),
                                ({"required_response": {"answers": [item]}}, "")]:
            self.assertEqual(parse_answers(json.dumps(payload)+suffix, {"q"})[0], [item])

    def test_reject_ambiguous_and_unexpected(self):
        answer = {"question_id": "q", "expression": "1+2"}
        bad = [{"answers": [answer, answer]}, {"answers": [answer], "required_response": {"answers": [answer]}},
               {"answers": [{"question_id": "foreign", "expression": "1+2"}]},
               {"answers": [{"question_id": "q", "expression": "<expression>"}]}]
        for payload in bad:
            with self.assertRaises(ValueError):
                parse_answers(json.dumps(payload), {"q"})
        for raw in ['{"answers":[],"answers":[]}', '{"answers":[]}}}', '{"answers":[]} prose']:
            with self.assertRaises(ValueError):
                parse_answers(raw)


if __name__ == "__main__":
    unittest.main()
