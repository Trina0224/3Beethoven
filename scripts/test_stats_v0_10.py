import unittest,itertools
from fractions import Fraction as F
from stats_curriculum_v0_10 import build,identity,calculate
from generate_stats_v0_10 import validate

class FractionCurriculumTests(unittest.TestCase):
    def test_exact_event_enumeration(self):
        for rows in build().values():
            for q in rows:
                if q['category']=='type_i':
                    n,r,a,d=q['parameters'];p=F(a,d)
                    truth=sum((p**sum(xs)*(1-p)**(n-sum(xs)) for xs in itertools.product((0,1),repeat=n) if sum(xs)==r),F(0))
                else:
                    a,b,d=q['parameters'];p1,p2=F(a,d),F(b,d)
                    truth=sum(((p1 if x else 1-p1)*(p2 if y else 1-p2) for x,y in itertools.product((0,1),repeat=2) if x+y==1),F(0))
                self.assertEqual(F(q['answer']),truth)
                self.assertEqual(q['choices']['ABCD'.index(q['answer_letter'])],q['answer'])
                validate(dict(stages=q['reference_chain'],answer=q['answer']),q)
    def test_normalized_identity(self):
        self.assertEqual(identity(dict(category='type_i',parameters=[5,2,3,20])),identity(dict(category='type_i',parameters=[5,2,6,40])))
        self.assertEqual(identity(dict(category='type_ii',parameters=[1,3,20])),identity(dict(category='type_ii',parameters=[6,2,40])))
    def test_reject_false_intermediate(self):
        q=build()['train'][0];obj=dict(stages=q['reference_chain'][:],answer=q['answer']);obj['stages'][1]='0'
        with self.assertRaises(AssertionError):validate(obj,q)
    def test_no_code_in_numeric_stage(self):
        with self.assertRaises(ValueError):calculate('__import__("os").getcwd()')

if __name__=='__main__':unittest.main()
