import copy
import json
import random
import unittest
import tempfile
from fractions import Fraction as F
from pathlib import Path

from exact_calculator import calculate
from stats_consolidation_grader import score, validate_raw_expression
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
               dict(kind="uniform", lower="-2", upper="10", cutoff="3", conditional=False),
               dict(kind="uniform", lower="-2", upper="10", cutoff="-2", conditional="false"),
               dict(kind="process", target="variance", rate="-1", duration="3"),
               dict(kind="moments", target="variance", variance="-5"),
               dict(kind="binomial", n="4", r="5", p="1/2")]
        for spec in bad:
            with self.assertRaises(ValueError):
                oracle(spec)

    def test_unconditional_uniform_uses_full_support_and_allows_negative_lower(self):
        spec = dict(kind="uniform", lower="-2", upper="10", cutoff="-2", conditional=False)
        self.assertEqual(oracle(spec), F(4))

    def test_conditional_uniform_uses_cutoff_not_lower(self):
        spec = dict(kind="uniform", lower="-2", upper="10", cutoff="4", conditional=True)
        self.assertEqual(oracle(spec), F(7))

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

    def test_whole_raw_training_contract_is_separate_from_math_primary(self):
        q = self.one("binomial")
        expression = q["expression"]
        for raw in (
            f"999;{expression}",
            f"x=999;{expression}",
            f"Expression: {expression} # trailing prose",
            "Expression: " + expression.replace("**", "^"),
            "Expression: " + expression.replace("comb", "C"),
            f"```\nExpression: {expression}\n```",
        ):
            judged = score(raw, q)
            self.assertTrue(judged["historical_executable_after_extraction"], raw)
            self.assertFalse(judged["whole_raw_executable"], raw)
            self.assertFalse(judged["strict_one_line_expression"], raw)

        self.assertEqual(validate_raw_expression(expression)[0], expression)
        self.assertTrue(score(expression, q)["primary_correct"])
        self.assertTrue(score(expression, q)["whole_raw_executable"])
        self.assertFalse(score(expression, q)["strict_one_line_expression"])
        self.assertTrue(score("Expression: " + expression, q)["primary_correct"])
        self.assertTrue(score("Expression: " + expression, q)["whole_raw_executable"])
        self.assertTrue(score("Expression: " + expression, q)["strict_one_line_expression"])

    def test_binomial_boundary_identities_are_credited_without_answer_only_relaxation(self):
        cases = [
            (0, "(1-(2/7))**12"),
            (1, "12*(2/7)*(1-(2/7))**11"),
            (11, "12*(2/7)**11*(1-(2/7))"),
            (12, "(2/7)**12"),
        ]
        for r, expression in cases:
            q = {
                "id": f"binomial_boundary_{r}",
                "category": "binomial",
                "question": f"X is binomial with n=12 and p=2/7; find P(X={r}).",
                "semantics": {"kind": "binomial", "n": "12", "r": str(r), "p": "2/7"},
                "bindings": {"n": "12", "r": str(r), "reject_probability": "2/7"},
                "expression": f"comb(12,{r})*(2/7)**{r}*(1-(2/7))**(12-{r})",
            }
            q["answer"] = str(oracle(q["semantics"]))
            self.assertTrue(score("Expression: " + expression, q)["primary_correct"], r)
            self.assertFalse(score("Expression: " + q["answer"], q)["primary_correct"], r)

    def test_binomial_boundary_wrong_exponents_and_coefficients_stay_wrong(self):
        q = {
            "id": "binomial_boundary_negative_bank",
            "category": "binomial",
            "question": "X is binomial with n=12 and p=2/7; find P(X=0).",
            "semantics": {"kind": "binomial", "n": "12", "r": "0", "p": "2/7"},
            "bindings": {"n": "12", "r": "0", "reject_probability": "2/7"},
            "expression": "comb(12,0)*(2/7)**0*(1-(2/7))**(12-0)",
        }
        q["answer"] = str(oracle(q["semantics"]))
        for wrong in ("(1-(2/7))**11", "12*(1-(2/7))**12", "(2/7)**12"):
            judged = score("Expression: " + wrong, q)
            self.assertFalse(judged["primary_correct"], wrong)
            self.assertFalse(judged["review_required"], wrong)

    def test_standard_event_expansions_are_preapproved_not_false_zeroes(self):
        p, z = "2/7", "3/11"
        canonical = {
            "both": f"({p})*({z})",
            "neither": f"(1-({p}))*(1-({z}))",
            "at_least_one": f"1-(1-({p}))*(1-({z}))",
            "exactly_one": f"({p})*(1-({z}))+(1-({p}))*({z})",
            "same": f"({p})*({z})+(1-({p}))*(1-({z}))",
        }
        expanded = {
            "both": f"({p})*({z})",
            "neither": f"1-({p})-({z})+({p})*({z})",
            "at_least_one": f"({p})+({z})-({p})*({z})",
            "exactly_one": f"({p})+({z})-2*({p})*({z})",
            "same": f"1-({p})-({z})+2*({p})*({z})",
        }
        for category in canonical:
            semantics = {"kind": "events", "target": category, "p": p, "p_b": z}
            q = {
                "id": "event_expansion_" + category,
                "category": category,
                "question": f"Independent events have probabilities {p} and {z}; find {category}.",
                "semantics": semantics,
                "bindings": {},
                "expression": canonical[category],
                "answer": str(oracle(semantics)),
            }
            judged = score("Expression: " + expanded[category], q)
            self.assertTrue(judged["primary_correct"], (category, judged))
            self.assertFalse(judged["review_required"], category)

    def test_event_expansion_near_misses_remain_rejected(self):
        semantics = {"kind": "events", "target": "at_least_one", "p": "2/7", "p_b": "3/11"}
        q = {
            "id": "event_expansion_near_miss",
            "category": "at_least_one",
            "question": "Independent events have probabilities 2/7 and 3/11; find at least one.",
            "semantics": semantics,
            "bindings": {},
            "expression": "1-(1-(2/7))*(1-(3/11))",
            "answer": str(oracle(semantics)),
        }
        for wrong in ("(2/7)+(3/11)-2*(2/7)*(3/11)",
                      "(2/7)+(3/11)+(2/7)*(3/11)"):
            judged = score("Expression: " + wrong, q)
            self.assertFalse(judged["primary_correct"], wrong)
            self.assertFalse(judged["review_required"], wrong)

    def test_poisson_product_second_moment_is_preapproved(self):
        semantics = {"kind": "poisson", "target": "second_moment", "mean": "13/4"}
        q = {
            "id": "poisson_second_product",
            "category": "poisson_second",
            "question": "X is Poisson with mean 13/4. Find E[X**2].",
            "semantics": semantics,
            "bindings": {},
            "expression": "(13/4)+(13/4)**2",
            "answer": str(oracle(semantics)),
        }
        judged = score("Expression: (13/4)*((13/4)+1)", q)
        self.assertTrue(judged["primary_correct"], judged)
        self.assertFalse(score("Expression: " + q["answer"], q)["primary_correct"])

    def test_fully_expanded_affine_second_moment_is_preapproved(self):
        semantics = {
            "kind": "moments", "target": "second_moment", "mean": "-3",
            "variance": "5", "scale": "-2/3", "offset": "4/5",
        }
        q = {
            "id": "affine_second_fully_expanded",
            "category": "moment_second",
            "question": "E[X]=-3 and Var(X)=5. Let Y=(-2/3)X+4/5. Find E[Y**2].",
            "semantics": semantics,
            "bindings": {},
            "expression": "(-2/3)**2*5+((-2/3)*(-3)+(4/5))**2",
            "answer": str(oracle(semantics)),
        }
        expanded = "(-2/3)**2*5+(-2/3)**2*(-3)**2+2*(-2/3)*(4/5)*(-3)+(4/5)**2"
        judged = score("Expression: " + expanded, q)
        self.assertTrue(judged["primary_correct"], judged)
        wrong = score("Expression: " + expanded.replace("+2*", "-2*"), q)
        self.assertFalse(wrong["primary_correct"])
        self.assertFalse(wrong["review_required"])

    def test_red_team_exact_legal_equivalence_fixtures(self):
        p, z = "77/113", "11/113"
        event_cases = {
            "neither": (
                f"(1-({p}))*(1-({z}))",
                [f"1-({p})-({z})+({p})*({z})"],
                f"1-({p})-({z})",
            ),
            "exactly_one": (
                f"({p})*(1-({z}))+(1-({p}))*({z})",
                [f"({p})+({z})-2*({p})*({z})",
                 f"(({p})+({z})-({p})*({z}))-({p})*({z})"],
                f"({p})*(1-({z}))",
            ),
            "at_least_one": (
                f"1-(1-({p}))*(1-({z}))",
                [f"({p})+({z})-({p})*({z})"],
                f"({p})+({z})",
            ),
            "same": (
                f"({p})*({z})+(1-({p}))*(1-({z}))",
                [f"1-({p})-({z})+2*({p})*({z})"],
                f"({p})*({z})",
            ),
        }
        for category, (reference, legal, wrong) in event_cases.items():
            semantics = {"kind": "events", "target": category, "p": p, "p_b": z}
            q = {
                "id": "red_team_" + category,
                "category": category,
                "question": f"Independent events have probabilities {p} and {z}; find {category}.",
                "semantics": semantics,
                "bindings": {},
                "expression": reference,
                "answer": str(oracle(semantics)),
            }
            for expression in legal:
                judged = score("Expression: " + expression, q)
                self.assertTrue(judged["primary_correct"], (category, expression, judged))
                self.assertFalse(judged["review_required"], (category, expression))
            judged = score("Expression: " + wrong, q)
            self.assertFalse(judged["primary_correct"], (category, wrong))
            self.assertFalse(judged["review_required"], (category, wrong))

        semantics = {
            "kind": "moments", "target": "second_moment", "mean": "-12",
            "variance": "83", "scale": "-48/7", "offset": "9/11",
        }
        q = {
            "id": "red_team_affine_second",
            "category": "moment_second",
            "question": "E[X]=-12 and Var(X)=83. Let Y=(-48/7)X+9/11. Find E[Y**2].",
            "semantics": semantics,
            "bindings": {},
            "expression": "(-48/7)**2*83+((-48/7)*(-12)+(9/11))**2",
            "answer": str(oracle(semantics)),
        }
        legal = [
            "(-48/7)**2*(83+(-12)**2)+2*(-48/7)*(9/11)*(-12)+(9/11)**2",
            "(-48/7)**2*83+(-48/7)**2*(-12)**2+2*(-48/7)*1*(9/11)*(-12)+(9/11)**2",
        ]
        for expression in legal:
            judged = score("Expression: " + expression, q)
            self.assertTrue(judged["primary_correct"], (expression, judged))
            self.assertFalse(judged["review_required"], expression)
        wrong = score("Expression: ((-48/7)*(-12)+(9/11))**2", q)
        self.assertFalse(wrong["primary_correct"])
        self.assertFalse(wrong["review_required"])

        semantics = {"kind": "poisson", "target": "second_moment", "mean": "158/23"}
        q = {
            "id": "red_team_poisson_second",
            "category": "poisson_second",
            "question": "X is Poisson with mean 158/23. Find E[X**2].",
            "semantics": semantics,
            "bindings": {},
            "expression": "(158/23)+(158/23)**2",
            "answer": str(oracle(semantics)),
        }
        judged = score("Expression: (158/23)*((158/23)+1)", q)
        self.assertTrue(judged["primary_correct"], judged)
        wrong = score("Expression: (158/23)**2", q)
        self.assertFalse(wrong["primary_correct"])
        self.assertFalse(wrong["review_required"])

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
        config["execution_authorization"] = "offline_repairs_only"
        with self.assertRaisesRegex(RuntimeError, "Offline repairs"):
            assert_execution_released(config)
        config["execution_authorization"] = "paid_execution_released"
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

    def test_offline_regrade_repairs_only_redundant_trailing_brace(self):
        from regrade_stats_consolidation_pilot import audit
        stories, requests, _, _ = build()
        request = next(r for r in requests if r["archetype"] == "chain_scaled_variance" and r["pilot_batch"])
        story = next(s for s in stories if s["story_id"] == request["story_id"])
        answers = [{"question_id": q["id"], "expression": q["expression"]}
                   for q in story["questions"]]
        record = {"request_sha256": digest(request), "accepted": {}, "attempts": [
            {"raw": json.dumps({"answers": answers}) + "}"}]}
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            (source/"records").mkdir()
            path = source/"records"/(request["story_id"]+".json")
            original = json.dumps(record)
            path.write_text(original)
            result = audit(stories, [request], source)
            self.assertEqual(result["classifications"], {"accepted": len(answers)})
            self.assertEqual(result["rows"][0]["attempts"][0]["transport_repair"],
                             "removed_redundant_trailing_closing_brace")
            self.assertEqual(path.read_text(), original)

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
