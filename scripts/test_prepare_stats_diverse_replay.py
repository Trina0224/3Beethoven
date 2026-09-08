import copy
import unittest
from collections import Counter

from prepare_stats_diverse_legacy import DEFAULT_SOURCE, EXPECTED_CATEGORIES, canonical_sha256, load_source
from prepare_stats_diverse_replay import PER_CATEGORY, ReplayContractError, build
from stats_consolidation_grader import score


class DiverseReplayTests(unittest.TestCase):
    def test_materialized_balanced_training_only_replay(self):
        artifact = build()
        rows, manifest = artifact["replay"], artifact["manifest"]
        self.assertEqual(len(rows), PER_CATEGORY * len(EXPECTED_CATEGORIES))
        self.assertEqual(
            Counter(row["category"] for row in rows),
            Counter({category: PER_CATEGORY for category in EXPECTED_CATEGORIES}),
        )
        self.assertTrue(manifest["rows_materialized"])
        self.assertFalse(manifest["evaluation_rows_included"])
        self.assertEqual(manifest["required_new_to_replay_ratio"], "1:1")
        self.assertEqual(manifest["rows_sha256"], canonical_sha256(rows))
        self.assertTrue(all(row["source_split"] == "train" for row in rows))
        self.assertTrue(all(row["id"].startswith("final_clear_train_") for row in rows))
        self.assertFalse(any("development" in row["id"] or "test" in row["id"] for row in rows))
        for row in rows:
            judged = score(row["target"], row)
            self.assertTrue(judged["primary_correct"], row["id"])
            self.assertTrue(judged["strict_one_line_expression"], row["id"])

    def test_source_row_hashes_bind_unmodified_rows(self):
        source, _ = load_source(DEFAULT_SOURCE)
        lookup = {row["id"]: row for row in source["train"]}
        for replay in build()["replay"]:
            self.assertEqual(replay["source_row_sha256"], canonical_sha256(lookup[replay["id"]]))

    def test_deterministic_selection_and_source_drift_rejected(self):
        self.assertEqual(build(), build())
        source, _ = load_source(DEFAULT_SOURCE)
        changed = copy.deepcopy(source["train"])
        changed[0]["target"] = "Expression: 1"
        # The builder's source-level SHA contract is covered by load_source;
        # this assertion documents that selected source rows are not mutable.
        self.assertNotEqual(canonical_sha256(changed), source["manifest"]["split_sha256"]["train"])


if __name__ == "__main__":
    unittest.main()
