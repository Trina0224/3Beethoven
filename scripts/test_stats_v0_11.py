import unittest,copy
from fractions import Fraction as F
from stats_curriculum_v0_11 import build,validate,score,make

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=build()
    def test_deterministic_disjoint_and_exact(self):
        self.assertEqual(self.data,build());validate(self.data)
    def test_known_operations(self):
        for k,p,a in [('multiply',(7,12,5,18),'35/216'),('add',(7,12,5,18),'31/36'),('integer_power',(17,1,3),'4913'),('fraction_power',(7,12,3),'343/1728'),('reduce',(17,31,19),'17/31')]:
            self.assertEqual(make(k,p,'x')['answer'],a)
    def test_score_not_expression_or_wrong_final(self):
        for raw in ['Calculation: 1/3*1/2','Calculation: 1/6 = 1/7\nAnswer: 1/7','Answer: 1/0','Answer: 1/6\nAnswer: 1/7','Answer: 0.1667']:
            self.assertFalse(score(raw,'1/6')['reviewed_correct'])
        self.assertTrue(score('Calculation: 1/3*1/2 = 1/6','1/6')['reviewed_correct'])
        self.assertTrue(score('Answer: 2/12','1/6')['correct'])
        self.assertFalse(score('Answer: 2/12','1/6','reduce')['correct'])
    def test_scaled_fractions_same_group(self):
        a=make('multiply',(1,2,2,3),'a');b=make('multiply',(2,4,4,6),'b')
        self.assertEqual(a['identity'],b['identity'])
        self.assertEqual(a['answer'],b['answer'])
    def test_leak_injection_rejected(self):
        d=copy.deepcopy(self.data);d['validation'][0]=d['train'][0]
        with self.assertRaises(AssertionError):validate(d)

if __name__=='__main__':unittest.main()
