import unittest
from stats_curriculum_v0_17 import build,score,selection_key

class CurriculumTests(unittest.TestCase):
 def test_split_and_correct_formulation(self):
  data=build();self.assertEqual([len(data[s]) for s in ('validation','test')],[48,96])
  self.assertFalse({tuple(q['identity']) for q in data['validation']} & {tuple(q['identity']) for q in data['test']})
  for rows in data.values():
   for q in rows:
    self.assertTrue(score('Expression: '+q['expression'],q)['correct'])
    self.assertFalse(score('Expression: 0',q)['correct'])
 def test_no_numeric_only_credit(self):
  for q in build()['validation']:
   if q['category'] in ('moment','poisson_scaled'):
    self.assertFalse(score('Expression: '+q['answer'],q)['correct'])
    b=q['bindings'];wrong=f"{b['scale']}**2*{b['mean']}+{b['offset']}"
    self.assertFalse(score('Expression: '+wrong,q)['correct'])
 def test_evaluated_square(self):
  for q in build()['validation']:
   if q['category'] in ('moment','poisson_scaled'):
    a=q['bindings']['scale'];expr=q['expression'].replace(a+'**2',str(int(a)**2),1)
    self.assertTrue(score('Expression: '+expr,q)['correct'])
 def test_exact_delta_mixture(self):
  try:import torch
  except ImportError:self.skipTest('Torch test executes on Kaggle')
  from run_stats_v0_17 import mixed_state
  a={'x.lora_A.weight':torch.tensor([[1.,2.]]),'x.lora_B.weight':torch.tensor([[3.],[4.]])}
  b={'x.lora_A.weight':torch.tensor([[5.,6.]]),'x.lora_B.weight':torch.tensor([[7.],[8.]])}
  for alpha in (0,.25,.5,.75,1):
   m=mixed_state(a,b,alpha)
   self.assertTrue(torch.allclose(m['x.lora_B.weight']@m['x.lora_A.weight'],(1-alpha)*(a['x.lora_B.weight']@a['x.lora_A.weight'])+alpha*(b['x.lora_B.weight']@b['x.lora_A.weight'])))

if __name__=='__main__':unittest.main()
