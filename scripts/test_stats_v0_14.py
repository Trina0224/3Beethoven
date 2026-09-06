import unittest
from stats_curriculum_v0_14 import build,score,digest


class TransferTests(unittest.TestCase):
    def test_frozen_and_disjoint(self):
        data=build()
        self.assertEqual(digest(data),'78da1ed6f18c5068e6dd0cc2608be16292c9eaa67c98d3864ca1db7691430d38')
        ids=[{tuple(q['identity']) for q in rs} for rs in data.values()]
        self.assertFalse(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
        for rs in data.values():
            for q in rs:self.assertTrue(score(q['target'],q)['correct'])

    def test_event_and_total_wait(self):
        data=build()
        for rs in data.values():
            for q in rs:
                if q['category']=='exactly_one':self.assertIn('one method detects and the other misses',q['question'])
                if q['category']=='at_least_one':self.assertIn('one or both methods detect',q['question'])
                if q['category']=='uniform_time':self.assertIn('total',q['question'].lower())


if __name__=='__main__':unittest.main()
