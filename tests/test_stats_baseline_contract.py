import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from stats_baseline_contract import baseline_manifest, validate_baseline_manifest


class BaselineCacheContractTests(unittest.TestCase):
    def setUp(self):
        self.metrics = {'old': {'correct': 34, 'n': 48}}
        self.context = {'parent_sha256': 'parent-a', 'selection_sha256': 'questions-a'}
        self.manifest = baseline_manifest(self.metrics, self.context)

    def test_matching_cache_is_accepted(self):
        validate_baseline_manifest(self.metrics, self.manifest, self.context)

    def test_legacy_or_different_numerical_configuration_is_rejected(self):
        for manifest in (None, {**self.manifest, 'numerical_preparation': 'unprepared'}):
            with self.assertRaises(RuntimeError):
                validate_baseline_manifest(self.metrics, manifest, self.context)

    def test_changed_weights_questions_or_scores_are_rejected(self):
        for context in ({**self.context, 'parent_sha256': 'parent-b'},
                        {**self.context, 'selection_sha256': 'questions-b'}):
            with self.assertRaises(RuntimeError):
                validate_baseline_manifest(self.metrics, self.manifest, context)
        with self.assertRaises(RuntimeError):
            validate_baseline_manifest({'old': {'correct': 35, 'n': 48}}, self.manifest, self.context)


if __name__ == '__main__':
    unittest.main()
