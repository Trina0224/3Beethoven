"""Regression checks for full-family replay, grading and retention gates."""
import unittest
from fractions import Fraction
from stats_curriculum_v0_19 import build,score,digest,KINDS
from run_stats_v0_19 import DATA_SHA,eligible

class ReplayRepair(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.data=build()
 def test_frozen_replay_coverage(self):
  d=self.data
  self.assertEqual(digest(d),DATA_SHA)
  self.assertEqual(len(d['train']),480)
  self.assertEqual(len({r['source_id'] for r in d['train']}),480)
  for k in KINDS:
   rows=[r for r in d['train'] if r['category']==k]
   self.assertEqual(len(rows),24)
   self.assertTrue(all(r.get('teacher_raw') for r in rows))
  train_ids={r['source_id'] for r in d['train']}
  self.assertFalse(train_ids & {q['id'] for k in d if k!='train' for q in d[k]})
 def test_equivalence_and_wrong_concept(self):
  d=self.data
  for q in d['old_test']+d['new_test']:
   self.assertTrue(score('Expression: '+q['expression'],q)['correct'])
   self.assertFalse(score('Expression: '+str(Fraction(q['answer'])+1),q)['correct'])
  q=next(q for q in d['old_test'] if q['category']=='binomial')
  self.assertTrue(score('Expression: '+q['expression'].replace('comb(', 'C('),q)['correct'])
  q=next(q for q in d['old_test'] if q['category']=='at_least_one')
  self.assertTrue(score('Expression: '+q['expression'].replace('(','').replace(')',''),q)['correct'])
  q=next(q for q in d['old_test'] if q['category']=='moment')
  self.assertFalse(score('Expression: (6*X+20)**2',q)['correct'])
 def test_each_family_gate_cannot_be_hidden_by_total(self):
  b={'correct':30,'by_category':{'a':6,'b':6}}
  self.assertTrue(eligible({'correct':28,'by_category':{'a':5,'b':5}},b))
  self.assertFalse(eligible({'correct':31,'by_category':{'a':4,'b':7}},b))
  self.assertFalse(eligible({'correct':27,'by_category':{'a':5,'b':5}},b))
if __name__=='__main__':unittest.main()
