import unittest
from fractions import Fraction as F
from stats_curriculum_v0_20 import build,score,EVENTS
from exact_calculator import calculate
class CurriculumTest(unittest.TestCase):
 def test_split_and_event_semantics(self):
  d=build();seen=set()
  for split,n in [('train',192),('validation',64),('test',96)]:
   self.assertEqual(len(d[split]),n)
   keys={tuple(sorted(q['parameters'][:2])) for q in d[split]}
   self.assertFalse(keys&seen);seen|=keys
   for q in d[split]:
    a,b,den=q['parameters'];p,r=F(a,den),F(b,den)
    # Independently enumerate four Bernoulli outcomes rather than reuse formulas.
    probs={(x,y):(p if x else 1-p)*(r if y else 1-r) for x in (0,1) for y in (0,1)}
    accept={'exactly_one':lambda x,y:x!=y,'both':lambda x,y:x+y==2,'neither':lambda x,y:x+y==0,'same':lambda x,y:x==y}[q['category']]
    expected=sum(v for (x,y),v in probs.items() if accept(x,y))
    self.assertEqual(F(calculate(q['expression'])),expected)
    self.assertTrue(score(q['target'],q)['correct'])
    if q['category']=='exactly_one':
     wrong=f'Expression: ({a}/101)*({b}/101)+(1-({a}/101))*(1-({b}/101))'
     self.assertFalse(score(wrong,q)['correct'])
if __name__=='__main__':unittest.main()
