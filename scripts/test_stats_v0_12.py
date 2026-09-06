import copy,unittest
from stats_curriculum_v0_12 import build,validate,score,make_micro,make_stats

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=build()
    def test_deterministic_and_disjoint(self):
        self.assertEqual(self.data,build());validate(self.data)
    def test_known_targets(self):
        self.assertEqual(make_micro('multiply_steps',(347,26),'x',0)['answer'],'9022')
        self.assertEqual(make_micro('power_steps',(17,3),'x',0)['answer'],'4913')
        self.assertEqual(make_micro('gcd_steps',(1071,462),'x',0)['answer'],'21')
        self.assertEqual(make_micro('reduce_steps',(84,126),'x',0)['answer'],'2/3')
        self.assertEqual(make_stats('type_i_pipeline',(4,2,7,50),'x',0)['answer'],'271803/3125000')
        self.assertEqual(make_stats('type_ii_pipeline',(7,11,50),'x',0)['answer'],'373/1250')
    def test_strict_score(self):
        self.assertTrue(score('Calculation: 2/6\nAnswer: 1/3','1/3')['correct'])
        for raw in ('Calculation: 1/3','Answer: 2/6','Answer: 1/3\nAnswer: 1/3','Answer: 0.3333'):
            self.assertFalse(score(raw,'1/3')['correct'])
    def test_leak_injection_rejected(self):
        d=copy.deepcopy(self.data);d['validation'][0]=d['train'][0]
        with self.assertRaises(AssertionError):validate(d)

if __name__=='__main__':unittest.main()
