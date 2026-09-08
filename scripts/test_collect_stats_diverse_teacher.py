"""Offline, full-size mock tests for the diverse teacher collector."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

import collect_stats_diverse_teacher as collector


def _fraction(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def make_curriculum(path: Path) -> tuple[dict[str, str], dict[str, object]]:
    expressions: dict[str, str] = {}
    document: dict[str, object] = {
        "manifest": {"fixture": True, "final_blind_sha256": "f" * 64},
    }
    counts = {"train": 48, "development": 6}
    split_offsets = {"train": 0, "development": 10000}
    for split, per_category in counts.items():
        rows = []
        for category_index in range(18):
            for index in range(per_category):
                serial = split_offsets[split] + category_index * 100 + index
                p = Fraction(serial + 2, serial + 1003)
                q = Fraction(serial + 3, serial + 2007)
                expression = f"({p.numerator}/{p.denominator})*({q.numerator}/{q.denominator})"
                answer = _fraction(p * q)
                qid = f"{split}_category_{category_index:02d}_{index:03d}"
                question = (
                    f"Two independent events have probabilities {p.numerator}/{p.denominator} and "
                    f"{q.numerator}/{q.denominator}. Find the probability that both occur. [{qid}]"
                )
                row = {
                    "id": qid,
                    "category": collector.EXPECTED_CATEGORIES[category_index],
                    "question": question,
                    "prompt": "Return only an expression for: " + question,
                    "semantics": {"kind": "events", "target": "both", "p": str(p), "p_b": str(q)},
                    "bindings": ({"lower": "0", "upper": "1", "width_divisor": "2"}
                                 if collector.EXPECTED_CATEGORIES[category_index] == "interval" else {}),
                    "expression": expression,
                    "answer": answer,
                    "target": "Expression: " + expression,
                    "verification": {
                        "prompt_semantics_verified": True,
                        "symbolic_oracle_verified": True,
                        "mutation_tests_passed": True,
                    },
                }
                if split == "train":
                    row.update(teacher_eligible=True, training_eligible=True, teacher_exposed=False)
                    row["teacher_binding"] = {
                        "result_id": "teacher_result_" + qid,
                        "training_row_id": qid,
                        "request_prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
                        "canonical_target_sha256": hashlib.sha256(row["target"].encode()).hexdigest(),
                        "collection_status": "pending_external_train_only_collection",
                        "required_release_state": "accepted_or_oracle_corrected",
                        "required_review_required": False,
                    }
                rows.append(row)
                expressions[qid] = expression
        document[split] = rows
    document["manifest"].update({
        "counts": {"train": 864, "development": 108, "final_blind": 144},
        "categories": list(collector.EXPECTED_CATEGORIES),
        "split_sha256": {
            "train": collector.digest(document["train"]),
            "development": collector.digest(document["development"]),
        },
        "isolation": {
            "teacher_allowed_input_splits": ["train"],
            "development_used_for_training": False,
            "final_blind_used_for_training": False,
        },
        "final_blind_commitment": {"artifact_file_sha256": "f" * 64},
        "verification_contract": {
            "all_rows_verified": True,
            "grader_sha256": collector.grader_fingerprint(),
            "builder_checker_source_sha256": collector.file_sha256(
                Path(collector.__file__).with_name("prepare_stats_diverse_curriculum.py")
            ),
        },
    })
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return expressions, document


class MockTeacher:
    def __init__(self, expressions: dict[str, str], bad_once: str | None = None,
                 bad_always: str | None = None, equivalent: str | None = None,
                 costs: list[object] | None = None):
        self.expressions = expressions
        self.bad_once = bad_once
        self.bad_always = bad_always
        self.equivalent = equivalent
        self.costs = list(costs or [])
        self.requests: list[dict[str, object]] = []
        self.seen: dict[str, int] = {}

    def __call__(self, request: dict[str, object], api_key: str) -> dict[str, object]:
        if api_key != "fixture-key":
            raise AssertionError("unexpected fixture key")
        self.requests.append(request)
        body = json.loads(request["messages"][1]["content"])
        answers = []
        for question in body["questions"]:
            qid = question["id"]
            count = self.seen.get(qid, 0)
            self.seen[qid] = count + 1
            expression = self.expressions[qid]
            if qid == self.bad_always or (qid == self.bad_once and count == 0):
                expression = "0"
            elif qid == self.equivalent:
                left, right = expression.split("*", 1)
                expression = right + "*" + left
            answers.append({"question_id": qid, "expression": expression})
        cost = self.costs.pop(0) if self.costs else 0.001
        usage = {} if cost == "missing" else {"cost": cost, "prompt_tokens": 10, "completion_tokens": 10}
        return {
            "raw": json.dumps({"answers": answers}, separators=(",", ":")),
            "finish_reason": "stop",
            "model": collector.MODEL,
            "provider": "MockProvider",
            "usage": usage,
            "response_id": f"mock-{len(self.requests)}",
        }


class DiverseTeacherCollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        cls.curriculum = cls.base / "curriculum.json"
        cls.expressions, cls.document = make_curriculum(cls.curriculum)
        cls.curriculum_sha = collector.file_sha256(cls.curriculum)
        # This deliberately invalid sealed artifact proves that the collector
        # neither requires nor opens a neighboring final-blind file.
        (cls.base / "STATS_DIVERSE_FINAL_BLIND.json").write_text("not valid JSON", encoding="utf-8")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    @staticmethod
    def fixture_verification():
        return mock.patch.object(
            collector,
            "_canonical_receipt",
            return_value={
                "prompt_semantics_verified": True,
                "symbolic_oracle_verified": True,
                "mutation_verified": True,
            },
        )

    def test_full_mock_retry_correction_no_leak_and_resume(self):
        root = self.base / "complete"
        train = self.document["train"]
        bad_once = train[0]["id"]
        bad_always = train[1]["id"]
        equivalent = train[2]["id"]
        teacher = MockTeacher(
            self.expressions,
            bad_once=bad_once,
            bad_always=bad_always,
            equivalent=equivalent,
        )

        with self.fixture_verification():
            paused = collector.run_collection(
                self.curriculum, root, "fixture-key", teacher, max_new_calls=1,
                expected_curriculum_sha256=self.curriculum_sha,
            )
        self.assertFalse(paused["training_release"])
        self.assertEqual(paused["usage"]["calls"], 1)
        self.assertEqual(len(teacher.requests), 1)
        with self.fixture_verification():
            result = collector.run_collection(
                self.curriculum, root, "fixture-key", teacher,
                expected_curriculum_sha256=self.curriculum_sha,
            )

        self.assertTrue(result["training_release"])
        self.assertEqual(result["counts"]["usable_targets"], 864)
        self.assertEqual(result["counts"]["raw_teacher_accepted"], 863)
        self.assertEqual(result["counts"]["canonical_corrected"], 1)
        self.assertEqual(result["counts"]["pending"], 0)
        self.assertEqual(
            result["supervision_description"],
            "hybrid_teacher_response_and_canonical_oracle_supervision",
        )
        self.assertEqual(result["supervision_proportions"]["raw_teacher_accepted"]["count"], 863)
        self.assertEqual(result["supervision_proportions"]["canonical_corrected"]["count"], 1)
        self.assertEqual(set(result["counts"]["by_category_status"]), set(collector.EXPECTED_CATEGORIES))
        for category_counts in result["counts"]["by_category_status"].values():
            self.assertEqual(
                set(category_counts),
                {"raw_teacher_accepted", "canonical_corrected", "pending", "failed_or_unattempted"},
            )
            self.assertEqual(sum(category_counts.values()), 48)
        self.assertEqual(result["usage"]["calls"], 73)
        self.assertEqual(len(teacher.requests), 73)
        self.assertEqual(
            (root / (collector.RESULT_NAME + ".sha256")).read_text().strip(),
            collector.file_sha256(root / collector.RESULT_NAME),
        )
        self.assertEqual(
            result["receipts"]["teacher_evidence_tree"]["file_count"],
            2 * result["usage"]["calls"],
        )
        tree = result["receipts"]["teacher_evidence_tree"]
        self.assertEqual(tree["file_count"], 146)
        self.assertRegex(tree["tree_sha256"], r"^[0-9a-f]{64}$")
        corrected = next(row for row in result["rows"] if row["training_row_id"] == bad_always)
        self.assertEqual(corrected["target_provenance"]["kind"], "canonical_corrected")
        self.assertEqual(corrected["training_target"], "Expression: " + self.expressions[bad_always])
        self.assertTrue(all(corrected["training_target_gates"].values()))
        alternative = next(row for row in result["rows"] if row["training_row_id"] == equivalent)
        self.assertEqual(alternative["status"], "raw_teacher_accepted")
        self.assertFalse(alternative["target_provenance"]["canonical_expression_exact"])
        self.assertNotEqual(alternative["training_target"], alternative["canonical_reference_target"])

        held_out = list(self.document["development"])
        forbidden = [row["id"] for row in held_out] + [row["question"] for row in held_out] + ["f" * 64]
        for request in teacher.requests:
            request_text = json.dumps(request, ensure_ascii=False)
            self.assertFalse(any(value in request_text for value in forbidden))
            self.assertTrue(request["provider"]["allow_fallbacks"])
            user = json.loads(request["messages"][1]["content"])
            self.assertEqual(set(user), {"questions"})
            self.assertTrue(all(set(item) == {"id", "question"} for item in user["questions"]))
        retry = teacher.requests[1]
        retry_body = json.loads(retry["messages"][1]["content"])
        self.assertEqual({q["id"] for q in retry_body["questions"]}, {bad_once, bad_always})
        self.assertIn("Generic rule card", retry["messages"][0]["content"])

        replay_guard = MockTeacher(self.expressions)
        with self.fixture_verification():
            replayed = collector.run_collection(
                self.curriculum, root, "fixture-key", replay_guard,
                expected_curriculum_sha256=self.curriculum_sha,
            )
        self.assertTrue(replayed["training_release"])
        self.assertEqual(replay_guard.requests, [])

    def test_missing_cost_stops_before_second_call_and_keeps_full_evidence(self):
        root = self.base / "missing-cost"
        teacher = MockTeacher(self.expressions, costs=["missing"])
        lookup_ids = []

        def unavailable_lookup(response_id, api_key):
            lookup_ids.append(response_id)
            raise RuntimeError("fixture metadata outage")

        with self.fixture_verification():
            result = collector.run_collection(
                self.curriculum, root, "fixture-key", teacher,
                provider_lookup=unavailable_lookup,
                expected_curriculum_sha256=self.curriculum_sha,
            )
        self.assertFalse(result["training_release"])
        self.assertEqual(result["usage"]["calls"], 1)
        self.assertEqual(result["usage"]["finalized_calls"], 0)
        self.assertEqual(result["usage"]["pending_paid_responses"], 1)
        self.assertTrue(result["usage"]["missing_cost"])
        self.assertEqual(len(teacher.requests), 1)
        self.assertEqual(lookup_ids, ["mock-1"])
        self.assertTrue((root / "inflight.json").exists())
        pending_path = next((root / "pending_responses").glob("*.json"))
        pending = json.loads(pending_path.read_text())
        self.assertEqual(pending["transport_response"]["response_id"], "mock-1")
        tree = result["receipts"]["teacher_evidence_tree"]
        self.assertEqual(tree["file_count"], 2)
        pending_entry = next(item for item in tree["files"] if item["path"].startswith("pending_responses/"))
        self.assertEqual(pending_entry["sha256"], collector.file_sha256(pending_path))

        def recovered_lookup(response_id, api_key):
            self.assertEqual(response_id, "mock-1")
            return {
                "response_body": {"data": {
                    "id": response_id,
                    "provider_name": "MockProvider",
                    "total_cost": 0.001,
                    "model": collector.MODEL,
                }},
                "data": {
                    "id": response_id,
                    "provider_name": "MockProvider",
                    "total_cost": 0.001,
                    "model": collector.MODEL,
                },
            }

        with self.fixture_verification():
            resumed = collector.run_collection(
                self.curriculum, root, "fixture-key", teacher,
                provider_lookup=recovered_lookup,
                expected_curriculum_sha256=self.curriculum_sha,
            )
        self.assertTrue(resumed["training_release"])
        self.assertEqual(resumed["usage"]["calls"], 72)
        self.assertEqual(len(teacher.requests), 72)
        self.assertFalse((root / "inflight.json").exists())
        self.assertFalse(pending_path.exists())

    def test_cost_cap_stops_immediately(self):
        root = self.base / "cost-cap"
        teacher = MockTeacher(self.expressions, costs=[0.301])
        with self.fixture_verification():
            result = collector.run_collection(
                self.curriculum, root, "fixture-key", teacher,
                expected_curriculum_sha256=self.curriculum_sha,
            )
        self.assertFalse(result["training_release"])
        self.assertEqual(result["usage"]["calls"], 1)
        self.assertGreater(result["usage"]["reported_cost_usd"], collector.MAX_COST_USD)
        self.assertEqual(len(teacher.requests), 1)

    def test_near_cap_reserves_next_request_before_transport(self):
        root = self.base / "near-cap"
        teacher = MockTeacher(self.expressions, costs=[0.299])
        with self.fixture_verification():
            result = collector.run_collection(
                self.curriculum, root, "fixture-key", teacher,
                expected_curriculum_sha256=self.curriculum_sha,
            )
        self.assertFalse(result["training_release"])
        self.assertEqual(result["usage"]["calls"], 1)
        self.assertEqual(result["usage"]["reported_cost_usd"], 0.299)
        self.assertEqual(len(teacher.requests), 1)
        self.assertIn("maximum-cost reservation", result["stop_reason"])

    def test_cached_bad_retry_still_hard_stops_before_new_call(self):
        root = self.base / "cached-bad-retry"
        bad = self.document["train"][0]["id"]
        first_teacher = MockTeacher(self.expressions, bad_always=bad, costs=[0.001, "missing"])
        def unavailable_lookup(response_id, api_key):
            raise RuntimeError("fixture metadata outage")

        with self.fixture_verification():
            first = collector.run_collection(
                self.curriculum, root, "fixture-key", first_teacher,
                provider_lookup=unavailable_lookup,
                expected_curriculum_sha256=self.curriculum_sha,
            )
        self.assertFalse(first["training_release"])
        self.assertEqual(first["usage"]["calls"], 2)
        self.assertEqual(first["usage"]["finalized_calls"], 1)
        self.assertEqual(first["usage"]["pending_paid_responses"], 1)

        replay_guard = MockTeacher(self.expressions)
        with self.fixture_verification():
            replayed = collector.run_collection(
                self.curriculum, root, "fixture-key", replay_guard,
                provider_lookup=unavailable_lookup,
                expected_curriculum_sha256=self.curriculum_sha,
            )
        self.assertFalse(replayed["training_release"])
        self.assertTrue(replayed["usage"]["missing_cost"])
        self.assertEqual(replay_guard.requests, [])

    def test_teacher_expression_contract_and_response_order_are_strict(self):
        row = self.document["train"][0]
        qid = row["id"]
        expression = row["expression"]

        def response(candidate):
            return json.dumps({"answers": [{"question_id": qid, "expression": candidate}]})

        _, accepted = collector.judge_response(response(expression), [row], "stop")
        self.assertTrue(accepted[0]["accepted"])
        malformed = [
            "Expression: " + expression,
            "999;" + expression,
            "x=999;" + expression,
            expression + " # explanation",
            "```" + expression + "```",
            expression.replace("*", "^", 1),
            "C(3,1)*1/2",
        ]
        for candidate in malformed:
            _, decisions = collector.judge_response(response(candidate), [row], "stop")
            self.assertFalse(decisions[0]["accepted"], candidate)
            self.assertFalse(decisions[0]["gates"]["whole_raw_executable"], candidate)

        _, partial = collector.parse_response(
            json.dumps({"answers": [{"question_id": "a", "expression": "1"}]}),
            ["a", "b"],
        )
        self.assertFalse(partial["strict_envelope"])
        _, reordered = collector.parse_response(
            json.dumps({"answers": [
                {"question_id": "b", "expression": "1"},
                {"question_id": "a", "expression": "1"},
            ]}),
            ["a", "b"],
        )
        self.assertFalse(reordered["strict_envelope"])

    def test_providerless_chat_response_uses_generation_receipt(self):
        root = self.base / "providerless"
        teacher = MockTeacher(self.expressions)

        def providerless_transport(request, api_key):
            response = teacher(request, api_key)
            response.pop("provider")
            return response

        def generation_lookup(response_id, api_key):
            data = {
                "id": response_id,
                "provider_name": "OfficialSchemaProvider",
                "total_cost": 0.001,
                "model": collector.MODEL,
            }
            return {"response_body": {"data": data}, "data": data}

        with self.fixture_verification():
            result = collector.run_collection(
                self.curriculum, root, "fixture-key", providerless_transport,
                provider_lookup=generation_lookup,
                max_new_calls=1,
                expected_curriculum_sha256=self.curriculum_sha,
            )
        self.assertFalse(result["training_release"])
        self.assertEqual(result["usage"]["calls"], 1)
        record = json.loads(next((root / "attempts").glob("*.json")).read_text())
        self.assertEqual(record["provider"], "OfficialSchemaProvider")
        self.assertEqual(record["provider_receipt"]["source"], "openrouter_generation_endpoint")
        self.assertEqual(record["response_id"], "mock-1")

    def test_manifest_is_portable_and_run_lock_is_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            left = base / "left" / "curriculum.json"
            right = base / "right" / "curriculum.json"
            left.parent.mkdir()
            right.parent.mkdir()
            payload = self.curriculum.read_bytes()
            left.write_bytes(payload)
            right.write_bytes(payload)
            document = json.loads(payload)
            self.assertEqual(
                collector._manifest(left, document, document["train"]),
                collector._manifest(right, document, document["train"]),
            )

            lock_root = base / "locked"
            with collector.exclusive_run_lock(lock_root):
                with self.assertRaisesRegex(RuntimeError, "holds this root lock"):
                    with collector.exclusive_run_lock(lock_root):
                        pass

    def test_curriculum_and_attempt_hash_tampering_are_rejected(self):
        root = self.base / "tamper"
        teacher = MockTeacher(self.expressions)
        with self.fixture_verification():
            first = collector.run_collection(
                self.curriculum, root, "fixture-key", teacher, max_new_calls=1,
                expected_curriculum_sha256=self.curriculum_sha,
            )
        self.assertFalse(first["training_release"])
        original = self.curriculum.read_text(encoding="utf-8")
        changed = json.loads(original)
        changed["manifest"]["tampered"] = True
        self.curriculum.write_text(json.dumps(changed) + "\n", encoding="utf-8")
        try:
            with self.assertRaisesRegex(RuntimeError, "pinned curriculum SHA-256"):
                with self.fixture_verification():
                    collector.run_collection(
                        self.curriculum, root, "fixture-key", MockTeacher(self.expressions),
                        expected_curriculum_sha256=self.curriculum_sha,
                    )
        finally:
            self.curriculum.write_text(original, encoding="utf-8")

        attempt_path = next((root / "attempts").glob("*.json"))
        attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        attempt["raw"] += " "
        attempt_path.write_text(json.dumps(attempt) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "evidence hash mismatch"):
            with self.fixture_verification():
                collector.validate_only(self.curriculum, root)

    def test_default_validation_does_not_read_key_or_call_transport(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(collector, "openrouter_transport", side_effect=AssertionError("network called")):
                with self.fixture_verification():
                    result = collector.validate_only(self.curriculum)
        self.assertTrue(result["curriculum_valid"])

    def test_real_generator_to_validate_only_integration(self):
        from prepare_stats_diverse_curriculum import (
            blind_artifact,
            build,
            canonical_json_text,
            public_artifact,
        )

        root = self.base / "real-generator"
        root.mkdir()
        public_path = root / "STATS_DIVERSE_CURRICULUM.json"
        blind_path = root / "STATS_DIVERSE_FINAL_BLIND.json"
        built = build(public_path, blind_path)
        public_path.write_text(canonical_json_text(public_artifact(built)), encoding="utf-8")
        blind_path.write_text(canonical_json_text(blind_artifact(built)), encoding="utf-8")
        # The public commitment is enough for collection. If validation opens
        # the separately sealed artifact, this intentionally invalid content
        # would make the test fail.
        blind_path.write_text("sealed and deliberately unreadable here", encoding="utf-8")
        with mock.patch.object(collector, "openrouter_transport", side_effect=AssertionError("network called")):
            result = collector.validate_only(public_path)
        self.assertTrue(result["curriculum_valid"])
        self.assertEqual(result["manifest"]["initial_packet_count"], 72)


if __name__ == "__main__":
    unittest.main()
