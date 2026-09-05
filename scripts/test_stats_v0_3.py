import json
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from stats_v0_3_common import (CONCEPTS, audit, digest, group_split, make_curriculum,
                             normalize_question, parse_answer, parse_teacher, prompt_for, read_frozen)
from flight_run_stats_v0_3 import TeacherClient, append, package, training_dataset


class DataTests(unittest.TestCase):
    def setUp(self):
        self.plan = make_curriculum()
        self.benchmark = read_frozen(Path(__file__).with_name("flight_run_stats_v0_2.py"))

    def test_balance_and_nonoverlap(self):
        result = audit(self.plan, self.benchmark)
        self.assertEqual(result["n"], 60)
        self.assertEqual(result["positions"], dict.fromkeys("ABCD", 15))
        self.assertEqual(len(self.benchmark), 24)
        self.assertEqual(Counter(r["answer_letter"] for r in self.benchmark), dict.fromkeys("ABCD", 6))

    def test_normalization_regression(self):
        self.assertEqual(normalize_question("A Poisson-variable?!"), "a poisson variable")
        self.assertEqual(normalize_question(" SAME, question! "), normalize_question("same question"))

    def test_deterministic(self):
        self.assertEqual(digest(self.plan), digest(make_curriculum()))

    def test_numeric_references(self):
        for ci in range(3):
            for v in range(10):
                r = self.plan[ci * 10 + v]
                expected = [13 + 2 * v, (2 + v % 4) * (13 + v) - (7 + v), 29 + 4 * v][ci]
                self.assertEqual(r["choices"]["ABCD".index(r["answer_letter"])], str(expected))

    def test_conceptual_references(self):
        for r in self.plan[30:]:
            chosen = r["choices"]["ABCD".index(r["answer_letter"])]
            if r["concept"] == "type_i_error":
                self.assertEqual(chosen, "Type I error")
            elif r["concept"] == "type_ii_error":
                self.assertEqual(chosen, "Type II error")
            else:
                v = int(r["id"].rsplit("_", 1)[1])
                self.assertEqual(chosen, "It becomes wider" if v % 2 == 0 else "It becomes narrower")

    def test_group_split(self):
        train, val = group_split(self.plan)
        self.assertEqual((len(train), len(val)), (48, 12))
        self.assertFalse({r["id"] for r in train} & {r["id"] for r in val})
        self.assertEqual(Counter(r["concept"] for r in val), dict.fromkeys(CONCEPTS, 2))

    def test_duplicate_gate(self):
        self.plan[1]["question"] = self.plan[0]["question"].upper() + "!"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit(self.plan, self.benchmark)

    def test_leakage_gate(self):
        self.plan[0]["question"] = self.benchmark[0]["question"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            audit(self.plan, self.benchmark)

    def test_parser_no_letter_fishing(self):
        for text, expected in [("B", "B"), ("Answer: D\nExplanation: test", "D"),
                               ("**Answer: C**", "C"), ("A. 13", "A"),
                               ("Consider A and B before answering", "INVALID"),
                               ("The correct answer is D", "INVALID"), ("AB", "INVALID"), ("", "INVALID")]:
            self.assertEqual(parse_answer(text), expected)

    def test_legacy_prompt(self):
        r = self.benchmark[0]
        self.assertTrue(prompt_for(r).endswith("Reply with ONLY the letter A, B, C, or D. Do not explain."))
        self.assertIn("D. 7", prompt_for(r))

    def test_paid_plaintext_response_recovery(self):
        raw = "Answer: B\nThe variance equals 15 because Poisson mean and variance are equal. A common mistake is to square the mean, which is incorrect."
        parsed = parse_teacher(raw)
        self.assertEqual(parsed["answer_letter"], "B")
        self.assertTrue(parsed["common_mistake"].startswith("A common mistake"))
        self.assertEqual(parse_teacher('```json\n{"answer_letter":"C"}\n```')["answer_letter"], "C")

    def test_training_masks_and_explicit_token_lists(self):
        calls = []
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                calls.append(kwargs)
                if kwargs.get("return_dict") is not False:
                    return {"input_ids": [10, 20, 30]}
                return [10, 20, 30] + ([40, 41] if len(messages) == 2 else [])
        fake_datasets = types.SimpleNamespace(Dataset=types.SimpleNamespace(from_list=lambda rows: rows))
        r = {**self.plan[0], "explanation": "Teacher explanation", "common_mistake": "Teacher misconception"}
        with patch.dict(sys.modules, {"datasets": fake_datasets}):
            rows = training_dataset([r], Tokenizer())
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(c["return_dict"] is False for c in calls))
        for row in rows:
            self.assertEqual(row["labels"], [-100, -100, -100, 40, 41])
            self.assertEqual(len(row["input_ids"]), len(row["attention_mask"]))

    def test_real_template_mismatch_still_stops(self):
        fake_datasets = types.SimpleNamespace(Dataset=types.SimpleNamespace(from_list=lambda rows: rows))
        tokenizer = types.SimpleNamespace(apply_chat_template=lambda messages, **kw: [1, 2] if len(messages) == 1 else [1, 3, 4])
        r = {**self.plan[0], "explanation": "Teacher explanation", "common_mistake": "Teacher misconception"}
        with patch.dict(sys.modules, {"datasets": fake_datasets}):
            with self.assertRaisesRegex(RuntimeError, "boundary mismatch"):
                training_dataset([r], tokenizer)


class LedgerTests(unittest.TestCase):
    def test_success_is_cached_and_capped(self):
        response = types.SimpleNamespace(status_code=200, json=lambda: {
            "choices": [{"message": {"content": "A"}}], "usage": {"cost": 0.001}})
        calls = []
        fake_requests = types.SimpleNamespace(post=lambda *a, **kw: calls.append(kw) or response,
                                             RequestException=OSError)
        with tempfile.TemporaryDirectory() as d, patch.dict(sys.modules, {"requests": fake_requests}):
            root = Path(d)
            client = TeacherClient(root, "test-only-not-a-secret", 1)
            messages = [{"role": "user", "content": "Example"}]
            self.assertEqual(client.call("one", messages), "A")
            resumed = TeacherClient(root, "test-only-not-a-secret", 1)
            self.assertEqual(resumed.call("one", messages), "A")
            self.assertEqual(len(calls), 1)
            self.assertEqual(resumed.stats()["attempted_calls"], 1)
            with self.assertRaisesRegex(RuntimeError, "cap"):
                resumed.call("two", messages)
            self.assertNotIn("test-only-not-a-secret", (root / "api_cache/one.json").read_text())
            self.assertEqual(resumed.stats()["reported_cost_usd"], 0.001)

    def test_inflight_is_not_rebilled(self):
        with tempfile.TemporaryDirectory() as d:
            client = TeacherClient(Path(d), "fake", 120)
            append(client.ledger, {"event": "started", "tag": "one"})
            with self.assertRaisesRegex(RuntimeError, "Unresolved"):
                client.call("one", [{"role": "user", "content": "test"}])

    def test_archive_integrity(self):
        import zipfile
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "run"
            root.mkdir()
            (root / "summary.json").write_text('{"ok":true}')
            package(root)
            with zipfile.ZipFile(Path(d) / "run.zip") as archive:
                self.assertEqual(set(archive.namelist()), {"summary.json", "manifest.json"})
                self.assertIsNone(archive.testzip())


if __name__ == "__main__":
    unittest.main()
