import unittest,itertools
from fractions import Fraction as F
from stats_curriculum_v0_9 import build,calculate
from generate_stats_v0_9 import validate

class ShortCalculationChecks(unittest.TestCase):
    def test_all_references_and_probability_enumeration(self):
        for rows in build().values():
            for q in rows:
                p=q['parameters'];answer=F(q['answer'])
                if q['category']=='type_i':
                    n,r,pn=p;prob=F(pn,20)
                    total=sum(prob**sum(bits)*(1-prob)**(n-sum(bits)) for bits in itertools.product((0,1),repeat=n) if sum(bits)==r)
                    self.assertEqual(total,answer)
                if q['category']=='type_ii':
                    b,c=[F(n,20) for n in p]
                    total=sum((b if first==0 else 1-b)*(c if second==0 else 1-c) for first,second in itertools.product((0,1),repeat=2) if first+second==1)
                    self.assertEqual(total,answer)
                validate(dict(rule=q['reference_rule'],calculation=q['reference_expression']+' = '+q['answer'],answer=q['answer']),q)

    def test_reject_false_intermediate_equality(self):
        q=build()['train'][0]
        with self.assertRaises(ValueError):validate(dict(rule=q['reference_rule'],calculation='1 = '+q['answer'],answer=q['answer']),q)

    def test_restricted_exact_arithmetic(self):
        self.assertEqual(calculate('0.1+0.2'),F(3,10))
        for expr in ('__import__("os")','2**9999','1/0','abs(2)'):
            with self.assertRaises((ValueError,ZeroDivisionError)):calculate(expr)

if __name__=='__main__':unittest.main()
