import unittest
from stats_training_accounting import invocation_loss


class InvocationLossTests(unittest.TestCase):
    def test_restart_boundaries_preserve_weighted_mean(self):
        # Independent sums for three invocations: 12, 12, and 9 updates.
        sums = (2.4, 1.2, 0.9)
        ranges = ((0, 12), (12, 24), (24, 33))
        records = [invocation_loss(s / end, start, end)
                   for s, (start, end) in zip(sums, ranges)]
        self.assertAlmostEqual(records[0]['training_loss'], .2)
        self.assertAlmostEqual(records[1]['training_loss'], .1)
        self.assertAlmostEqual(records[2]['training_loss'], .1)
        weighted = sum(r['training_loss'] * r['invocation_updates'] for r in records) / 33
        self.assertAlmostEqual(weighted, sum(sums) / 33)

    def test_invalid_progress_or_loss_is_rejected(self):
        for loss, start, end in ((.1, 12, 12), (.1, 24, 12),
                                 (float('nan'), 0, 12), (-.1, 0, 12)):
            with self.assertRaises(ValueError):
                invocation_loss(loss, start, end)

    def test_unverified_framework_is_not_silently_corrected(self):
        record = invocation_loss(.03, 24, 33, 'future-version')
        self.assertIsNone(record['training_loss'])
        self.assertEqual(record['raw_trainer_training_loss'], .03)


if __name__ == '__main__':
    unittest.main()
