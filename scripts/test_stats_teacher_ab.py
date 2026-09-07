import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from stats_teacher_ab import DATA, read_json, messages, judge, intermediates, summarize, call, save_json, append


class ABTests(unittest.TestCase):
    def test_all_reference_fixtures_and_step_errors(self):
        for q in read_json(DATA)["questions"]:
            answer = {"answers": [{"question_id": q["id"], "expression": q["expression"]}]}
            self.assertTrue(judge(json.dumps(answer), q, "A")["accepted"])
            answer["intermediates"] = {k:str(v) for k,v in intermediates(q).items()}
            self.assertTrue(judge(json.dumps(answer), q, "B")["accepted"])
            answer["intermediates"][next(iter(answer["intermediates"]))] = "-999"
            self.assertFalse(judge(json.dumps(answer), q, "B")["accepted"])
            self.assertEqual(json.loads(messages(q, "A")[1]["content"]), json.loads(messages(q, "B")[1]["content"]))
            self.assertNotIn("expression", json.loads(messages(q, "A")[1]["content"]))

    def test_per_family_gate_and_incomplete_run(self):
        rows = [{"id":q["id"], "family":q["development_family"], "arm":a,
                 "accepted":True, "classification":"accepted"} for q in read_json(DATA)["questions"] for a in ("A","B")]
        self.assertEqual(summarize(rows, True)["selected_arm"], "A")
        self.assertIsNone(summarize(rows, False)["selected_arm"])
        for r in rows[:4]:
            r["accepted"] = False
        self.assertIsNone(summarize(rows, True)["selected_arm"])

    def test_unresolved_or_missing_cost_never_calls(self):
        for history in ([{"event":"started", "tag":"old"}],
                        [{"event":"started", "tag":"old"}, {"event":"response", "tag":"old", "usage":{}}]):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for record in history:
                    append(root / "api_ledger.jsonl", record)
                fake = Mock()
                with patch.dict("sys.modules", {"requests": fake}):
                    with self.assertRaises(RuntimeError):
                        call(root, "unused", "new", [{"role":"user", "content":"test"}])
                    fake.post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
