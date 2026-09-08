import copy
import unittest

from prepare_stats_diverse_legacy import (
    DEFAULT_SOURCE,
    EXPECTED_CATEGORIES,
    LegacyContractError,
    SPLITS,
    build,
    load_source,
    validate_split,
)


class LegacyRetentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document, _ = load_source(DEFAULT_SOURCE)

    def test_real_source_is_exact_and_all_references_pass(self):
        manifest = build(DEFAULT_SOURCE)
        self.assertEqual(manifest["categories"], list(EXPECTED_CATEGORIES))
        self.assertEqual(manifest["splits"]["development"]["count"], 72)
        self.assertEqual(manifest["splits"]["final"]["count"], 144)
        self.assertTrue(manifest["validation"]["all_216_reference_targets_accepted"])
        self.assertEqual(manifest["validation"]["pending_equivalences"], 0)
        self.assertFalse(manifest["validation"]["rows_duplicated_into_this_manifest"])

    def test_reordered_rows_are_rejected(self):
        rows = copy.deepcopy(self.document["development"])
        rows[0], rows[1] = rows[1], rows[0]
        with self.assertRaisesRegex(LegacyContractError, "canonical SHA-256 drift"):
            validate_split(rows, "development", SPLITS["development"])

    def test_reference_or_semantics_edit_is_rejected(self):
        rows = copy.deepcopy(self.document["test"])
        rows[0]["target"] = "Expression: 1"
        with self.assertRaisesRegex(LegacyContractError, "canonical SHA-256 drift"):
            validate_split(rows, "final", SPLITS["final"])


if __name__ == "__main__":
    unittest.main()
