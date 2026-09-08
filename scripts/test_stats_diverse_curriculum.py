import copy
import json
import re
import tempfile
import unittest
from collections import Counter, defaultdict
from fractions import Fraction as F
from pathlib import Path
from unittest import mock

from exact_calculator import calculate
from prepare_stats_diverse_curriculum import (
    CATEGORIES,
    CATEGORY_TO_FAMILY,
    DEFAULT_BLIND_OUTPUT,
    DEFAULT_OUTPUT,
    EVENT_CATEGORIES,
    HELD_OUT_STRUCTURAL_COMBINATIONS,
    SPLIT_SIZES,
    SUFFIX,
    axis_audit,
    blind_artifact,
    build,
    canonical_json_text,
    canonical_key,
    canonical_semantics,
    category_axis_audit,
    digest,
    discriminative_audit,
    expected_misconception_audit,
    historical_registry,
    load_public_curriculum,
    normalize_template,
    ordering_audit,
    parse_question_semantics,
    public_artifact,
    qstr,
    story_projection,
    template_signature,
    text_digest,
    text_registry_audit,
    trap_audit,
    validate_prompt_semantics,
    verify_row,
)
from stats_consolidation_grader import score
from stats_consolidation_semantics import oracle, spec_for, task_key


def manual_question(category, question, semantics, expression):
    return {
        "id": "manual_" + category,
        "category": category,
        "question": question,
        "prompt": question + SUFFIX,
        "semantics": semantics,
        "expression": expression,
        "answer": str(oracle(semantics)),
        "target": "Expression: " + expression,
        "bindings": (
            {
                "lower": semantics["lower"],
                "upper": semantics["upper"],
                "width_divisor": semantics["divisor"],
            }
            if category == "interval"
            else {}
        ),
    }


class DiverseCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.public = json.loads(DEFAULT_OUTPUT.read_text())
        cls.blind = json.loads(DEFAULT_BLIND_OUTPUT.read_text())
        cls.data = {
            "train": cls.public["train"],
            "development": cls.public["development"],
            "final_blind": cls.blind["final_blind"],
            "manifest": cls.public["manifest"],
        }

    def test_physical_split_counts_and_commitment(self):
        self.assertEqual(set(self.public), {"train", "development", "manifest"})
        self.assertEqual(set(self.blind), {"final_blind", "receipt"})
        expected_totals = {split: per_category * 18 for split, per_category in SPLIT_SIZES.items()}
        self.assertEqual(self.data["manifest"]["counts"], expected_totals)
        for split, total in expected_totals.items():
            rows = self.data[split]
            self.assertEqual(len(rows), total)
            self.assertEqual(Counter(row["category"] for row in rows), Counter({c: SPLIT_SIZES[split] for c in CATEGORIES}))
            self.assertEqual(digest(rows), self.data["manifest"]["split_sha256"][split])
        receipt = self.blind["receipt"]
        receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        self.assertEqual(digest(receipt_body), receipt["receipt_sha256"])
        commitment = self.data["manifest"]["final_blind_commitment"]
        self.assertEqual(commitment["path"], "docs/STATS_DIVERSE_FINAL_BLIND.json")
        self.assertEqual(commitment["count"], 144)
        self.assertEqual(commitment["split_sha256"], digest(self.blind["final_blind"]))
        self.assertEqual(commitment["receipt_sha256"], receipt["receipt_sha256"])
        self.assertEqual(commitment["artifact_file_sha256"], text_digest(DEFAULT_BLIND_OUTPUT.read_text()))
        streams = self.data["manifest"]["generation_streams"]
        self.assertEqual(set(streams), set(SPLIT_SIZES))
        self.assertEqual(len({item["split_domain_label_sha256"] for item in streams.values()}), 3)
        self.assertEqual(
            receipt["generation_stream_sha256"],
            streams["final_blind"]["split_domain_label_sha256"],
        )

    def test_deterministic_rebuild_and_views(self):
        rebuilt = build(DEFAULT_OUTPUT, DEFAULT_BLIND_OUTPUT)
        self.assertEqual(public_artifact(rebuilt), self.public)
        self.assertEqual(blind_artifact(rebuilt), self.blind)

    def test_build_bytes_are_independent_of_runtime_output_paths(self):
        with tempfile.TemporaryDirectory() as left_dir, tempfile.TemporaryDirectory() as right_dir:
            left = build(Path(left_dir) / "public.json", Path(left_dir) / "blind.json")
            right = build(Path(right_dir) / "different-public.json", Path(right_dir) / "different-blind.json")
        self.assertEqual(canonical_json_text(public_artifact(left)), canonical_json_text(public_artifact(right)))
        self.assertEqual(canonical_json_text(blind_artifact(left)), canonical_json_text(blind_artifact(right)))

    def test_discarded_31415_fuzz_build(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch(
            "prepare_stats_diverse_curriculum.SEED", 31415
        ):
            fuzz = build(Path(temp_dir) / "public.json", Path(temp_dir) / "blind.json")
        self.assertEqual(fuzz["manifest"]["seed"], 31415)
        self.assertEqual(ordering_audit(fuzz["train"])["event_rows_per_microbatch_histogram"], {"2": 84, "3": 24})
        self.assertEqual(len(fuzz["train"]), 864)
        self.assertEqual(len(fuzz["development"]), 108)
        self.assertEqual(len(fuzz["final_blind"]), 144)
        contract = self.data["manifest"]["non_selection_fuzz_contract"]
        self.assertFalse(contract["eligible_for_training"])
        self.assertFalse(contract["eligible_for_model_or_seed_selection"])
        self.assertFalse(contract["artifact_persisted"])

    def test_every_row_repasses_independent_parser_oracle_and_grader(self):
        for split in SPLIT_SIZES:
            for row in self.data[split]:
                before = copy.deepcopy(row)
                self.assertEqual(F(calculate(row["expression"])), oracle(row["semantics"]))
                self.assertTrue(score(row["target"], row)["primary_correct"])
                verify_row(row)
                self.assertEqual(row, before, row["id"])
                self.assertTrue(row["verification"]["independent_prompt_parser_passed"])
                self.assertTrue(row["verification"]["independent_oracle_correct"])
                self.assertTrue(row["verification"]["grader_primary_correct"])

    def test_independent_parser_rejects_numeric_and_target_phrase_mutations(self):
        one_per_category = {}
        for row in self.data["development"]:
            one_per_category.setdefault(row["category"], row)
        self.assertEqual(set(one_per_category), set(CATEGORIES))
        for category, original in one_per_category.items():
            row = copy.deepcopy(original)
            match = re.search(r"\((-?\d+(?:/\d+)?)\)", row["question"])
            self.assertIsNotNone(match, category)
            replacement = qstr(F(match.group(1)) + 1)
            row["question"] = row["question"][: match.start(1)] + replacement + row["question"][match.end(1) :]
            row["prompt"] = row["question"] + SUFFIX
            row["template_signature"] = template_signature(row["question"], row["prompt"])
            with self.assertRaises(ValueError, msg=category):
                validate_prompt_semantics(row)
        event = copy.deepcopy(one_per_category["exactly_one"])
        event["question"] = event["question"].replace("exactly one of A and B occurs", "at least one of A and B occurs")
        event["prompt"] = event["question"] + SUFFIX
        event["template_signature"] = template_signature(event["question"], event["prompt"])
        with self.assertRaises(ValueError):
            validate_prompt_semantics(event)
        moment = copy.deepcopy(one_per_category["moment_mean"])
        moment["question"] = moment["question"].replace("Find E[Y].", "Find Var(Y).")
        moment["prompt"] = moment["question"] + SUFFIX
        moment["template_signature"] = template_signature(moment["question"], moment["prompt"])
        with self.assertRaises(ValueError):
            validate_prompt_semantics(moment)
        uniform = copy.deepcopy(one_per_category["uniform_mean"])
        uniform["semantics"]["cutoff"] = qstr(F(uniform["semantics"]["lower"]) + 1)
        with self.assertRaises(ValueError):
            validate_prompt_semantics(uniform)

    def test_hand_authored_parser_contract_for_all_18_categories(self):
        cases = {}
        for category, phrase in {
            "both": "both A and B occur",
            "neither": "neither A nor B occurs",
            "exactly_one": "exactly one of A and B occurs",
            "at_least_one": "at least one of A and B occurs",
            "same": "both occur or neither occurs",
        }.items():
            cases[category] = (
                f"A and B are independent events. P(B)=(3/11); P(A)=(2/7). Find the probability that {phrase}.",
                {"kind": "events", "target": category, "p": "2/7", "p_b": "3/11"},
            )
        for category, target, ask in (
            ("moment_mean", "mean", "E[Y]"),
            ("moment_variance", "variance", "Var(Y)"),
            ("moment_second", "second_moment", "E[Y**2]"),
        ):
            cases[category] = (
                f"E[X]=(-3/2); Var(X)=(5/7). Define Y=(-4/3)*X+(2/5). Find {ask}.",
                {"kind": "moments", "target": target, "mean": "-3/2", "variance": "5/7", "scale": "-4/3", "offset": "2/5"},
            )
        cases.update(
            {
                "poisson_variance": (
                    "X has a Poisson distribution with mean (5/2). Find Var(X).",
                    {"kind": "poisson", "target": "variance", "mean": "5/2"},
                ),
                "poisson_scaled": (
                    "X has a Poisson distribution with mean (5/2). Define Y=(-3/2)*X+(4/7). Find Var(Y).",
                    {"kind": "poisson", "target": "variance", "mean": "5/2", "scale": "-3/2", "offset": "4/7"},
                ),
                "poisson_second": (
                    "X has a Poisson distribution with mean (5/2). Find E[X**2].",
                    {"kind": "poisson", "target": "second_moment", "mean": "5/2"},
                ),
                "process_variance": (
                    "A homogeneous Poisson process has rate (3/2) arrivals per minute. X counts arrivals during (90) seconds. Find Var(X).",
                    {"kind": "process", "target": "variance", "rate": "3/2", "duration": "3/2"},
                ),
                "process_scaled": (
                    "A homogeneous Poisson process has rate (3/2) arrivals per minute. X counts arrivals during (7/3) minutes. Define Y=(-2/3)*X+(5/11). Find Var(Y).",
                    {"kind": "process", "target": "variance", "rate": "3/2", "duration": "7/3", "scale": "-2/3", "offset": "5/11"},
                ),
                "process_second": (
                    "A homogeneous Poisson process has rate (3/2) arrivals per minute. X counts arrivals during (4) minutes. Find E[X**2].",
                    {"kind": "process", "target": "second_moment", "rate": "3/2", "duration": "4"},
                ),
                "uniform_mean": (
                    "T is uniform on [(1/2), (9/2)] minutes. Find E[T] in minutes.",
                    {"kind": "uniform", "conditional": False, "lower": "1/2", "cutoff": "1/2", "upper": "9/2"},
                ),
                "uniform_conditional": (
                    "T is uniform on [(1/2), (9/2)] minutes. Given T>(90) seconds, find the conditional mean of total T in minutes.",
                    {"kind": "uniform", "conditional": True, "lower": "1/2", "cutoff": "3/2", "upper": "9/2"},
                ),
                "binomial": (
                    "X is the number of successes in (7) independent trials, each with success probability (2/5). Find P(X=(0)).",
                    {"kind": "binomial", "n": "7", "r": "0", "p": "2/5"},
                ),
                "interval": (
                    "A normal-theory confidence interval is [(-5/2), (7/3)]. The sample size is multiplied by (9/4); the center, confidence level, and population standard deviation stay fixed. Find the new upper endpoint.",
                    {"kind": "interval", "lower": "-5/2", "upper": "7/3", "divisor": "3/2"},
                ),
            }
        )
        self.assertEqual(set(cases), set(CATEGORIES))
        for category, (question, expected) in cases.items():
            parsed = parse_question_semantics(question, category, "direct_shared")
            self.assertEqual(canonical_semantics(parsed), canonical_semantics(expected), category)

    def test_cross_split_and_historical_collision_contracts(self):
        for field in ("semantic_key", "group_story_key", "question", "prompt"):
            registries = {split: {canonical_key(row[field]) if not isinstance(row[field], str) else row[field] for row in self.data[split]} for split in SPLIT_SIZES}
            self.assertFalse(registries["train"] & registries["development"], field)
            self.assertFalse(registries["train"] & registries["final_blind"], field)
            self.assertFalse(registries["development"] & registries["final_blind"], field)
        historical_tasks, historical_stories, historical_questions, historical_prompts, audit = historical_registry((DEFAULT_OUTPUT, DEFAULT_BLIND_OUTPUT))
        self.assertIn("docs/STATS_FINAL_CLEAR_DATA.json.gz.b64", audit["source_files"])
        all_rows = [row for split in SPLIT_SIZES for row in self.data[split]]
        self.assertFalse({canonical_key(row["semantic_key"]) for row in all_rows} & historical_tasks)
        self.assertFalse(
            {
                canonical_key(story_projection(spec_for(row)))
                for row in all_rows
            }
            & historical_stories
        )
        self.assertFalse({row["question"] for row in all_rows} & historical_questions)
        self.assertFalse({row["prompt"] for row in all_rows} & historical_prompts)

    def test_coverage_slots_and_contrast_groups(self):
        train, development, final = (self.data[name] for name in ("train", "development", "final_blind"))
        events = [row for row in train if row["category"] in EVENT_CATEGORIES]
        for rows in (train, development, final):
            split_events = [row for row in rows if row["category"] in EVENT_CATEGORIES]
            self.assertEqual(
                {row["coverage_axes"]["denominator_relation"] for row in split_events},
                {"same", "different"},
            )
            self.assertEqual(
                {row["coverage_axes"]["given_order"] for row in split_events},
                {"A_then_B", "B_then_A"},
            )
        self.assertIn("zero", {row["coverage_axes"]["probability_a_position"] for row in events} | {row["coverage_axes"]["probability_b_position"] for row in events})
        self.assertIn("one", {row["coverage_axes"]["probability_a_position"] for row in events} | {row["coverage_axes"]["probability_b_position"] for row in events})
        for rows in (train, development):
            self.assertFalse(
                any(
                    row["coverage_axes"]["denominator_relation"] == "same"
                    and row["coverage_axes"]["given_order"] == "B_then_A"
                    for row in rows
                    if row["category"] in EVENT_CATEGORIES
                )
            )
        for rows in (development, final):
            event_groups = defaultdict(list)
            for row in rows:
                if row["category"] in EVENT_CATEGORIES:
                    event_groups[row["contrast_group"]].append(row)
            for group in event_groups.values():
                self.assertEqual({row["category"] for row in group}, set(EVENT_CATEGORIES))
                p, p_b = F(group[0]["semantics"]["p"]), F(group[0]["semantics"]["p_b"])
                self.assertTrue(0 < p < 1 and 0 < p_b < 1)
                self.assertNotEqual(p, p_b)
                self.assertNotEqual(p + p_b, 1)
                self.assertEqual(len({F(row["answer"]) for row in group}), 5)
        moment_rows = [row for row in train if row["category"] in ("moment_mean", "moment_variance", "moment_second")]
        self.assertIn("negative_fraction", {row["coverage_axes"]["scale_form"] for row in moment_rows})
        self.assertIn("negative_fraction", {row["coverage_axes"]["offset_form"] for row in moment_rows})
        self.assertIn("zero", {row["coverage_axes"]["scale_form"] for row in moment_rows})
        self.assertIn("zero", {row["coverage_axes"]["offset_form"] for row in moment_rows})
        poisson_rows = [row for row in train if row["family"] == "poisson"]
        self.assertIn("positive_fraction", {row["coverage_axes"]["mean_form"] for row in poisson_rows})
        process_rows = [row for row in train if row["family"] == "process"]
        self.assertEqual({row["coverage_axes"]["duration_unit"] for row in process_rows}, {"seconds", "minutes"})
        self.assertIn("positive_fraction", {row["coverage_axes"]["rate_form"] for row in process_rows})
        for split in SPLIT_SIZES:
            conditional = [row for row in self.data[split] if row["category"] == "uniform_conditional"]
            self.assertEqual({row["coverage_axes"]["cutoff_unit"] for row in conditional}, {"seconds", "minutes"})
            self.assertTrue(any(row["coverage_axes"]["lower_form"] == "positive_fraction" for row in conditional))
            r_slots = {row["coverage_axes"]["r_slot"] for row in self.data[split] if row["category"] == "binomial"}
            self.assertEqual(r_slots, {"r_zero", "r_one", "r_interior", "r_n_minus_one", "r_n"})
            probability_regions = {
                row["coverage_axes"]["probability_region"]
                for row in self.data[split]
                if row["category"] == "binomial"
            }
            self.assertEqual(probability_regions, {"above_half", "below_half"})
            multipliers = {row["coverage_axes"]["sample_multiplier_form"] for row in self.data[split] if row["category"] == "interval"}
            self.assertIn("positive_fraction", multipliers)
        train_uniform = [row for row in train if row["category"] == "uniform_conditional"]
        self.assertTrue(
            any(
                row["coverage_axes"]["cutoff_unit"] == "seconds"
                and row["coverage_axes"]["lower_form"] == "positive_integer"
                for row in train_uniform
            )
        )
        self.assertTrue(
            any(
                row["coverage_axes"]["cutoff_unit"] == "minutes"
                and row["coverage_axes"]["lower_form"] == "positive_fraction"
                for row in train_uniform
            )
        )
        binomial_train_pairs = {
            (row["coverage_axes"]["n_parity"], row["coverage_axes"]["probability_region"])
            for row in train
            if row["category"] == "binomial"
        }
        self.assertTrue(
            {("even", "below_half"), ("odd", "below_half"), ("odd", "above_half")}
            <= binomial_train_pairs
        )
        for rows in (train, development):
            self.assertNotIn(
                ("even", "above_half"),
                {
                    (row["coverage_axes"]["n_parity"], row["coverage_axes"]["probability_region"])
                    for row in rows
                    if row["category"] == "binomial"
                },
            )
        final_high_even = [
            row
            for row in final
            if row["category"] == "binomial"
            and row["coverage_axes"]["n_parity"] == "even"
            and row["coverage_axes"]["probability_region"] == "above_half"
        ]
        self.assertEqual({row["coverage_axes"]["r_slot"] for row in final_high_even}, {"r_zero", "r_one", "r_interior", "r_n_minus_one", "r_n"})
        self.assertEqual(len({row["contrast_group"] for row in final_high_even}), 1)

        self.assertEqual({row["coverage_axes"]["endpoint_form"] for row in train if row["category"] == "interval"}, {"integers", "contains_fraction"})
        train_interval_pairs = Counter(
            (row["coverage_axes"]["interval_geometry"], row["coverage_axes"]["endpoint_form"])
            for row in train
            if row["category"] == "interval"
        )
        self.assertGreater(train_interval_pairs[("negative", "integers")], 0)
        self.assertEqual(train_interval_pairs[("negative", "contains_fraction")], 0)
        for geometry in ("cross_zero", "positive", "zero_lower"):
            self.assertGreater(train_interval_pairs[(geometry, "integers")], 0)
            self.assertGreater(train_interval_pairs[(geometry, "contains_fraction")], 0)
        self.assertEqual({row["coverage_axes"]["interval_geometry"] for row in development if row["category"] == "interval"}, {"negative", "cross_zero", "positive"})
        self.assertEqual({row["coverage_axes"]["interval_geometry"] for row in final if row["category"] == "interval"}, {"negative", "cross_zero", "positive", "zero_lower"})
        for rows in (development, final):
            self.assertEqual(
                {row["coverage_axes"]["endpoint_form"] for row in rows if row["category"] == "interval"},
                {"integers", "contains_fraction"},
            )
        self.assertFalse(
            any(
                row["coverage_axes"]["interval_geometry"] == "negative"
                and row["coverage_axes"]["endpoint_form"] == "contains_fraction"
                for row in development
                if row["category"] == "interval"
            )
        )
        final_negative_fraction = [
            row
            for row in final
            if row["category"] == "interval"
            and row["coverage_axes"]["interval_geometry"] == "negative"
            and row["coverage_axes"]["endpoint_form"] == "contains_fraction"
        ]
        self.assertTrue(final_negative_fraction)
        self.assertIn(
            "positive_fraction",
            {row["coverage_axes"]["sample_multiplier_form"] for row in final_negative_fraction},
        )

    def test_axes_hashes_and_predeclared_final_combinations(self):
        manifest = self.data["manifest"]
        for split in SPLIT_SIZES:
            self.assertEqual(axis_audit(self.data[split]), manifest["coverage_axis_audit"][split])
            self.assertEqual(category_axis_audit(self.data[split]), manifest["category_coverage_axis_audit"][split])
        held = manifest["predeclared_held_out_structural_combinations"]
        self.assertEqual(set(held), set(HELD_OUT_STRUCTURAL_COMBINATIONS))
        matched = {}
        for name, item in held.items():
            self.assertEqual(item["counts"]["train"], 0)
            self.assertEqual(item["counts"]["development"], 0)
            self.assertGreater(item["counts"]["final_blind"], 0)
            requirements = HELD_OUT_STRUCTURAL_COMBINATIONS[name]
            matched[name] = [
                row
                for row in self.data["final_blind"]
                if all(
                    {"family": CATEGORY_TO_FAMILY[row["category"]], **row["coverage_axes"]}.get(key)
                    == value
                    for key, value in requirements.items()
                )
            ]
            self.assertTrue(matched[name])
            self.assertTrue(all(row["discriminative_mutations"] for row in matched[name]), name)

        event_rows = matched["event_same_denominator_b_then_a"]
        event_groups = defaultdict(list)
        for row in event_rows:
            event_groups[row["contrast_group"]].append(row)
        self.assertTrue(
            any(
                {row["category"] for row in group} == set(EVENT_CATEGORIES)
                and len({F(row["answer"]) for row in group}) == 5
                for group in event_groups.values()
            )
        )
        self.assertEqual(
            {row["coverage_axes"]["r_slot"] for row in matched["binomial_even_n_above_half"]},
            {"r_zero", "r_one", "r_interior", "r_n_minus_one", "r_n"},
        )
        self.assertTrue(
            all(
                len(row["discriminative_mutations"]) == len(row["misconception_traps"])
                for row in matched["interval_negative_fractional_rational_multiplier"]
            )
        )

    def test_training_interleave_microbatches_cycles_and_windows(self):
        train = self.data["train"]
        self.assertEqual(ordering_audit(train), self.data["manifest"]["ordering"]["microbatch_and_window_audit"])
        histogram = Counter()
        for start in range(0, len(train), 8):
            batch = train[start : start + 8]
            category_counts = Counter(row["category"] for row in batch)
            self.assertLessEqual(max(category_counts.values()), 2)
            family_counts = Counter(CATEGORY_TO_FAMILY[row["category"]] for row in batch)
            histogram[family_counts["events"]] += 1
            self.assertTrue(all(count <= 2 for family, count in family_counts.items() if family != "events"))
        self.assertEqual(histogram, Counter({2: 84, 3: 24}))
        for start in range(0, len(train), 18):
            self.assertEqual({row["category"] for row in train[start : start + 18]}, set(CATEGORIES))
        for start in range(0, len(train), 288):
            self.assertEqual(Counter(row["category"] for row in train[start : start + 288]), Counter({c: 16 for c in CATEGORIES}))

    def test_misconception_mutations_are_audited_not_training_targets(self):
        manifest = self.data["manifest"]
        for split in SPLIT_SIZES:
            rows = self.data[split]
            self.assertEqual(trap_audit(rows), manifest["misconception_trap_audit"][split])
            self.assertEqual(discriminative_audit(rows, split != "train"), manifest["discriminative_mutation_audit"][split])
            for row in rows:
                self.assertEqual(row["misconception_traps"], expected_misconception_audit(row))
                self.assertNotIn(row["target"].removeprefix("Expression: "), {trap["expression"] for trap in row["misconception_traps"]})
                for trap in row["misconception_traps"]:
                    self.assertEqual(F(trap["computed"]), F(calculate(trap["expression"])))
                    if not trap["collides_with_oracle"]:
                        self.assertFalse(score("Expression: " + trap["expression"], row)["primary_correct"])
        boundary = [
            (row, trap)
            for split in ("development", "final_blind")
            for row in self.data[split]
            if row["category"] == "binomial" and F(row["semantics"]["r"]) in (0, F(row["semantics"]["n"]))
            for trap in row["misconception_traps"]
            if trap["id"] == "omit_binomial_coefficient"
        ]
        self.assertTrue(boundary)
        self.assertTrue(all(trap["collides_with_oracle"] for _, trap in boundary))

    def test_public_loader_never_opens_blind_and_final_has_no_teacher_binding(self):
        opened = []
        original = Path.read_text

        def audited(path, *args, **kwargs):
            opened.append(Path(path).resolve())
            return original(path, *args, **kwargs)

        with mock.patch.object(Path, "read_text", audited):
            loaded = load_public_curriculum(DEFAULT_OUTPUT)
        self.assertEqual(set(loaded), {"train", "development", "manifest"})
        self.assertEqual(opened, [DEFAULT_OUTPUT.resolve()])
        self.assertNotIn(DEFAULT_BLIND_OUTPUT.resolve(), opened)
        self.assertTrue(all(row["teacher_binding"] is not None and row["teacher_eligible"] for row in self.data["train"]))
        self.assertTrue(all(row["teacher_binding"] is None and not row["teacher_eligible"] for row in self.data["development"] + self.data["final_blind"]))
        public_text = canonical_json_text(self.public)
        for row in self.data["final_blind"]:
            self.assertNotIn(row["id"], public_text)
            self.assertNotIn(row["group_story_id"], public_text)
            self.assertNotIn(canonical_key(row["semantic_key"]), public_text)

    def test_template_signature_normalizer_and_overlap_reporting(self):
        self.assertEqual(normalize_template("  X=(−3/4)  + 12 "), "x=(−<num>) + <num>")
        for split in SPLIT_SIZES:
            for row in self.data[split]:
                self.assertEqual(row["template_signature"], template_signature(row["question"], row["prompt"]))
        self.assertEqual(text_registry_audit({s: self.data[s] for s in SPLIT_SIZES}), self.data["manifest"]["exact_text_and_template_registry"])
        overlaps = self.data["manifest"]["exact_text_and_template_registry"]["cross_split_overlap"]
        self.assertTrue(all(item["exact_question_count"] == item["exact_prompt_count"] == 0 for item in overlaps.values()))
        self.assertTrue(any(item["shared_template_signature_count"] > 0 for item in overlaps.values()))

    def test_independent_legal_equivalence_bank(self):
        p, q = "2/7", "3/11"
        event_bank = {
            "both": (
                "both A and B occur",
                f"({p})*({q})",
                f"({p})*({q})",
                f"(1-({p}))*(1-({q}))",
            ),
            "neither": (
                "neither A nor B occurs",
                f"(1-({p}))*(1-({q}))",
                f"1-({p})-({q})+({p})*({q})",
                f"1-({p})-({q})",
            ),
            "exactly_one": (
                "exactly one of A and B occurs",
                f"({p})*(1-({q}))+(1-({p}))*({q})",
                f"({p})+({q})-2*({p})*({q})",
                f"({p})+({q})-({p})*({q})",
            ),
            "at_least_one": (
                "at least one of A and B occurs",
                f"1-(1-({p}))*(1-({q}))",
                f"({p})+({q})-({p})*({q})",
                f"({p})+({q})",
            ),
            "same": (
                "both occur or neither occurs",
                f"({p})*({q})+(1-({p}))*(1-({q}))",
                f"1-({p})-({q})+2*({p})*({q})",
                f"({p})*({q})",
            ),
        }
        for category, (phrase, reference, legal, wrong) in event_bank.items():
            event = manual_question(
                category,
                f"A and B are independent events. P(A)=(2/7); P(B)=(3/11). Find the probability that {phrase}.",
                {"kind": "events", "target": category, "p": p, "p_b": q},
                reference,
            )
            self.assertTrue(score("Expression: " + legal, event)["primary_correct"], category)
            self.assertFalse(score("Expression: " + wrong, event)["primary_correct"], category)

        mean, variance, scale, offset = "-3/2", "5/7", "-4/3", "2/5"
        moment = manual_question(
            "moment_second",
            "E[X]=(-3/2); Var(X)=(5/7). Define Y=(-4/3)*X+(2/5). Find E[Y**2].",
            {"kind": "moments", "target": "second_moment", "mean": mean, "variance": variance, "scale": scale, "offset": offset},
            f"({scale})**2*({variance})+(({scale})*({mean})+({offset}))**2",
        )
        expanded = f"({scale})**2*(({variance})+({mean})**2)+2*({scale})*({offset})*({mean})+({offset})**2"
        self.assertTrue(score("Expression: " + expanded, moment)["primary_correct"])
        self.assertFalse(score(f"Expression: (({scale})*({mean})+({offset}))**2", moment)["primary_correct"])

        mu = "158/23"
        poisson = manual_question(
            "poisson_second",
            "X has a Poisson distribution with mean (158/23). Find E[X**2].",
            {"kind": "poisson", "target": "second_moment", "mean": mu},
            f"({mu})+({mu})**2",
        )
        self.assertTrue(score(f"Expression: ({mu})*(({mu})+1)", poisson)["primary_correct"])
        self.assertFalse(score(f"Expression: ({mu})**2", poisson)["primary_correct"])

        process = manual_question(
            "process_second",
            "A homogeneous Poisson process has rate (3/2) arrivals per minute. X counts arrivals during (90) seconds. Find E[X**2].",
            {"kind": "process", "target": "second_moment", "rate": "3/2", "duration": "3/2"},
            "((3/2)*(90/60))+((3/2)*(90/60))**2",
        )
        self.assertTrue(score("Expression: ((3/2)*(90/60))*(1+((3/2)*(90/60)))", process)["primary_correct"])
        self.assertFalse(score("Expression: ((3/2)*90)*(1+((3/2)*90))", process)["primary_correct"])

        uniform = manual_question(
            "uniform_conditional",
            "T is uniform on [(1/2), (9/2)] minutes. Given T>(90) seconds, find the conditional mean of total T in minutes.",
            {"kind": "uniform", "conditional": True, "lower": "1/2", "cutoff": "3/2", "upper": "9/2"},
            "((3/2)+(9/2))/2",
        )
        self.assertTrue(score("Expression: (3/2)+((9/2)-(3/2))/2", uniform)["primary_correct"])
        self.assertFalse(score("Expression: ((9/2)-(3/2))/2", uniform)["primary_correct"])

        interval = manual_question(
            "interval",
            "A normal-theory confidence interval is [(-5/2), (7/3)]. The sample size is multiplied by (9/4); the center, confidence level, and population standard deviation stay fixed. Find the new upper endpoint.",
            {"kind": "interval", "lower": "-5/2", "upper": "7/3", "divisor": "3/2"},
            "((-5/2)+(7/3))/2+((7/3)-(-5/2))/(2*(3/2))",
        )
        self.assertTrue(score("Expression: ((-5/2)+(7/3))/2+(((7/3)-(-5/2))/2)/(3/2)", interval)["primary_correct"])
        self.assertFalse(score("Expression: ((-5/2)+(7/3))/2+(((7/3)-(-5/2))/2)/(9/4)", interval)["primary_correct"])

        for r in (0, 1, 6, 7):
            semantics = {"kind": "binomial", "n": "7", "r": str(r), "p": "2/5"}
            reference = f"comb(7,{r})*(2/5)**{r}*(1-(2/5))**(7-{r})"
            row = manual_question(
                "binomial",
                f"X is the number of successes in (7) independent trials, each with success probability (2/5). Find P(X=({r})).",
                semantics,
                reference,
            )
            alternatives = {
                0: "(1-(2/5))**7",
                1: "7*(2/5)*(1-(2/5))**6",
                6: "7*(2/5)**6*(1-(2/5))",
                7: "(2/5)**7",
            }
            self.assertTrue(score("Expression: " + alternatives[r], row)["primary_correct"], r)
            near_wrong = {
                0: "(1-(2/5))**6",
                1: "(2/5)*(1-(2/5))**6",
                6: "(2/5)**6*(1-(2/5))",
                7: "(2/5)**6",
            }
            self.assertFalse(score("Expression: " + near_wrong[r], row)["primary_correct"], r)
        interior = manual_question(
            "binomial",
            "X is the number of successes in (7) independent trials, each with success probability (2/5). Find P(X=(3)).",
            {"kind": "binomial", "n": "7", "r": "3", "p": "2/5"},
            "comb(7,3)*(2/5)**3*(1-(2/5))**(7-3)",
        )
        self.assertFalse(score("Expression: (2/5)**3*(1-(2/5))**4", interior)["primary_correct"])


if __name__ == "__main__":
    unittest.main()
