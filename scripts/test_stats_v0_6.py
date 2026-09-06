import unittest
from fractions import Fraction as F
from stats_holdout_v0_6 import build,audit
from run_stats_rotation_v1 import rotate
from stats_curriculum_v0_5 import build as previous

class FrozenStoryTest(unittest.TestCase):
    def test_numeric_references(self):
        expected={'poisson':[9,16,15,24,21,32,27,40],
                  'expectation':[57,89,129,177,233,297,369,449],
                  'uniform':[9,12,15,18,21,24,27,30],
                  'confidence':[38,48,58,68,78,88,98,108]}
        for topic,answers in expected.items():
            rows=[r for r in build() if r['category']==topic]
            self.assertEqual([F(r['choices']['ABCD'.index(r['answer_letter'])]) for r in rows],answers)
    def test_event_enumeration(self):
        import itertools
        for topic in ('type_i','type_ii'):
            for v,r in enumerate(q for q in build() if q['category']==topic):
                n=3 if topic=='type_i' else 2
                p=F(v+1,20) if topic=='type_i' else 1-F(v+2,20)
                count=2 if topic=='type_i' else 1
                expected=sum(p**sum(bits)*(1-p)**(n-sum(bits)) for bits in itertools.product((0,1),repeat=n) if sum(bits)==count)
                self.assertEqual(F(r['choices']['ABCD'.index(r['answer_letter'])]),expected)
    def test_freeze_and_rotation_mapping(self):
        self.assertEqual(audit(sum(previous().values(),[]))['n'],48)
        for q in build():
            gold=q['choices']['ABCD'.index(q['answer_letter'])]
            for shift in range(4):
                r=rotate(q,shift)
                self.assertEqual(r['choices']['ABCD'.index(r['answer_letter'])],gold)
if __name__=='__main__':unittest.main()

