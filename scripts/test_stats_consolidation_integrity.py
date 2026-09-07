import copy
import json
import random
import unittest
import tempfile
from fractions import Fraction as F
from pathlib import Path

from exact_calculator import calculate
from stats_consolidation_grader import score
from stats_consolidation_holdout import sample_old_parameters
from stats_consolidation_pilot import build, selection_validation, assert_execution_released
from stats_consolidation_pilot import historical_replay_questions, digest
from stats_consolidation_semantics import oracle, validate_question, task_key, assert_disjoint
from stats_curriculum_v0_13 import make


class IntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stories = build()[0]
        cls.questions = [q for s in cls.stories for q in s["questions"]]

    def one(self, category):
        return copy.deepcopy(next(q for q in self.questions if q["category"] == category))

    def test_original_118_percent_bug_is_rejected_even_when_reference_is_echoed(self):
        q = make("binomial", [118, 1300, 4, 3], "regression", 0)
        with self.assertRaisesRegex(ValueError, "Probability"):
            score("Expression: " + q["expression"], q)

    def test_original_181_percent_scanners_are_rejected_for_both_targets(self):
        for category in ("exactly_one", "at_least_one"):
            q = make(category, [181, 1218, 4, 3], "regression", 0)
            with self.assertRaisesRegex(ValueError, "Probability"):
                validate_question(q)

    def test_text_probability_cannot_disagree_with_legal_metadata(self):
        q = self.one("binomial")
        q["question"] = "Each independent check has probability 118%."
        with self.assertRaisesRegex(ValueError, "prompt"):
            validate_question(q)

    def test_legal_but_inconsistent_prompt_probability_is_rejected(self):
        q = self.one("binomial")
        q["question"] = q["question"].replace("18%", "19%")
        with self.assertRaisesRegex(ValueError, "disagrees"):
            validate_question(q)

    def test_all_candidate_and_selection_questions_have_independent_reference_checks(self):
        qs = self.questions + [q for rows in selection_validation(self.stories)["suites"].values() for q in rows]
        for q in qs:
            validate_question(q)

    def test_wrong_reference_and_wrong_answer_together_do_not_pass(self):
        q = self.one("poisson_time")
        q["expression"] = "1"
        q["answer"] = "1"
        with self.assertRaisesRegex(ValueError, "independent oracle"):
            validate_question(q)

    def test_impossible_distribution_parameters(self):
        bad = [dict(kind="uniform", lower="0", upper="10", cutoff="10", conditional=True),
               dict(kind="process", target="variance", rate="-1", duration="3"),
               dict(kind="moments", target="variance", variance="-5"),
               dict(kind="binomial", n="4", r="5", p="1/2")]
        for spec in bad:
            with self.assertRaises(ValueError):
                oracle(spec)

    def test_semantic_keys_ignore_nuisance_story_parameters(self):
        q = self.one("v18_second_moment")
        q["semantics"] = dict(kind="poisson", target="second_moment", mean="72")
        other = copy.deepcopy(q)
        other.update(id="another_story", parameters=[999, 888])
        self.assertEqual(task_key(q), task_key(other))
        with self.assertRaisesRegex(ValueError, "Subtask leakage"):
            assert_disjoint({"train": [q], "validation": [other]})

    def test_unit_equivalent_subtasks_share_one_key(self):
        q = self.one("poisson_time")
        q["semantics"]["duration"] = "120/60"
        other = copy.deepcopy(q)
        other["semantics"]["duration"] = "2"
        self.assertEqual(task_key(q), task_key(other))

    def test_variance_key_ignores_constant_offset(self):
        q = self.one("poisson_scaled")
        other = copy.deepcopy(q)
        other["semantics"]["offset"] = "999"
        self.assertEqual(task_key(q), task_key(other))

    def test_candidate_splits_and_selection_are_disjoint_at_subtask_level(self):
        train = historical_replay_questions() + [q for s in self.stories if s["split"] == "train" for q in s["questions"]]
        validation = [q for s in self.stories if s["split"] == "validation" for q in s["questions"]]
        assert_disjoint({"train": train, "validation": validation})
        selected = [q for rows in selection_validation(self.stories)["suites"].values() for q in rows]
        assert_disjoint({"train": train, "selection": selected})

    def test_generators_keep_percentages_in_range_across_seeds(self):
        for seed in (2027, 31415, 1919):
            rng = random.Random(seed)
            for category in ("binomial", "exactly_one", "at_least_one"):
                for i in range(100):
                    q = make(category, sample_old_parameters(category, rng), "regression", i)
                    validate_question(q)
                    self.assertTrue(0 <= F(q["answer"]) <= 1)

    def test_original_teacher_binomial_equivalence_gets_credit(self):
        q = self.one("binomial")
        self.assertTrue(score("Expression: 9*(18/100)*(1-18/100)**(9-1)", q)["primary_correct"])

    def test_total_wait_equivalences_get_credit_but_remaining_wait_does_not(self):
        q = self.one("uniform_time")
        t, u = q["semantics"]["cutoff"], q["semantics"]["upper"]
        self.assertTrue(score(f"Expression: ({t})+({u}-({t}))/2", q)["primary_correct"])
        self.assertTrue(score(f"Expression: {u}-({u}-({t}))/2", q)["primary_correct"])
        self.assertFalse(score(f"Expression: ({u}-({t}))/2", q)["primary_correct"])

    def test_interval_other_midpoint_form_gets_credit(self):
        q = self.one("interval")
        b = q["bindings"]
        e = f"{b['upper']}-({b['upper']}-{b['lower']})/2+({b['upper']}-{b['lower']})/(2*{b['width_divisor']})"
        self.assertTrue(score("Expression: " + e, q)["primary_correct"])

    def test_final_number_alone_does_not_get_new_equivalence_credit(self):
        q = self.one("uniform_time")
        self.assertFalse(score("Expression: " + q["answer"], q)["primary_correct"])

    def test_wrong_unit_and_incomplete_symbolic_formula_stay_uncredited(self):
        q = self.one("poisson_time")
        self.assertFalse(score("Expression: ("+q["expression"]+")/60", q)["primary_correct"])
        q = self.one("poisson_scaled")
        self.assertFalse(score("Expression: 4*Var(X)", q)["primary_correct"])

    def test_config_blocks_paid_calls_without_changing_research_thresholds(self):
        config = json.loads((Path(__file__).resolve().parents[1]/"docs/STATS_CONSOLIDATION_DECISION_DRAFT.json").read_text())
        self.assertEqual(config["teacher"]["pilot_minimum_accepted_targets"], 59)
        with self.assertRaisesRegex(RuntimeError, "Offline repairs"):
            assert_execution_released(config)

    def test_offline_regrade_uses_raw_and_never_overwrites_source(self):
        from regrade_stats_consolidation_pilot import audit
        stories, requests, _, _ = build()
        request = next(r for r in requests if r["archetype"] == "binomial" and r["pilot_batch"])
        qid = request["questions"][0]["question_id"]
        # Deliberately synthetic fixture; not an actual pilot result.
        record = {"request_sha256": digest(request), "accepted": {}, "attempts": [{"raw": json.dumps(
            {"answers": [{"question_id": qid, "expression": "9*(18/100)*(1-18/100)**(9-1)"}]}),
            "parsed": {qid: "wrong cached parsed text"}}]}
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            (source/"records").mkdir()
            path = source/"records"/(request["story_id"]+".json")
            original = json.dumps(record)
            path.write_text(original)
            result = audit(stories, [request], source)
            self.assertEqual(result["newly_credited"], [qid])
            self.assertEqual(path.read_text(), original)
            self.assertEqual(result["teacher_calls_added"], 0)
            record["request_sha256"] = "stale"
            path.write_text(json.dumps(record))
            result = audit(stories, [request], source)
            self.assertEqual(result["classifications"], {"unavailable": 1})

    def test_readiness_rerun_preserves_actual_pilot_accounting(self):
        from check_stats_consolidation_readiness import assess
        prior = {"teacher_calls": 38, "gpu_runs": 0, "pilot": {"accepted_targets": 42,
                 "planned_targets": 65, "reported_cost_usd": 0.00402078, "gate_passed": False}}
        first = assess(prior, {"returncode": 0})
        second = assess(first, {"returncode": 0})
        self.assertEqual(second["teacher_calls"], 38)
        self.assertEqual(second["pilot"]["accepted_targets"], 42)
        self.assertEqual(second["pilot"]["reported_cost_usd"], 0.00402078)
        self.assertFalse(second["paid_execution_ready"])


if __name__ == "__main__":
    unittest.main()
