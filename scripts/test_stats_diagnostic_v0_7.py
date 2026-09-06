import unittest
from fractions import Fraction
from diagnose_stats_v0_7 import tasks,numeric,score

class DiagnosticScoring(unittest.TestCase):
    def test_no_number_fishing(self):
        for raw in ('2 or 3','Use 3 trials and probability 1/20','Answer: 1/0'):
            self.assertIsNone(numeric(raw))
        self.assertEqual(numeric('Calculation: 3*4\nAnswer: 12'),12)
        self.assertEqual(numeric('Answer: 1/4'),Fraction(1,4))

    def test_exact_mapping_and_task_coverage(self):
        rows=tasks()
        self.assertEqual(len(rows),288)
        self.assertEqual(len({r['id'] for r in rows}),24)
        for row in rows:
            answer=row['expected'] if row['mode'] in ('mc','mapping') else 'Answer: '+row['expected']
            self.assertTrue(score(answer,row)[1])
            self.assertTrue(score('unclear',row)[2])

    def test_probability_tolerance(self):
        row=next(r for r in tasks() if r['id']=='v06_type_i_00' and r['mode']=='free')
        # Three disjoint outcomes, each with probability .05*.05*.95.
        self.assertTrue(score('Answer: 0.007125',row)[1])
        self.assertFalse(score('Answer: 0.007',row)[1])

if __name__=='__main__': unittest.main()
