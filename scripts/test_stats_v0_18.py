import unittest
from fractions import Fraction as F
from stats_curriculum_v0_18 import build,make,score,stage_rows,TRACKS,digest

class CurriculumTests(unittest.TestCase):
    def test_frozen_split_and_reference_values(self):
        data=build();self.assertEqual(digest(data),'0e51ed04578b29e8d91f931dfbeec78a94867ba3d52dd6a45e06c45ee186f57c')
        seen={}
        for split,rows in data.items():
            for q in rows:
                a,m,c,b,v,u,t=q['parameters'];d=q['depth'];track=q['track']
                if track=='poisson_variance':expected=F(a) if d==1 else F(a*m) if d==2 else F(a*(m*60+30),60)
                elif track=='scaled_variance':expected=F(c*c*(v if d==1 else a if d==2 else a*m))
                elif track=='second_moment':expected=F(v+a*a) if d==1 else F(a+a*a) if d==2 else F(a*m+(a*m)**2)
                else:expected=(F(t) if d<3 else F(t*60+30,60))+u;expected=expected/2
                self.assertEqual(F(q['answer']),expected)
                key=(track,q['question']);self.assertEqual(seen.setdefault(key,split),split)
        for s in (1,2,3):
            rows=stage_rows(data,s)
            self.assertEqual({r['depth'] for r in rows},set(range(1,s+1)))
            for d in range(1,s+1):self.assertEqual(sum(r['depth']==d for r in rows),96 if d==s else 48)

    def test_misconceptions_not_credited(self):
        p=[17,4,3,7,23,40,5]
        q=make('scaled_variance',1,p,'probe',0)
        for wrong in ('3*23','3**2*23+7','23'):
            self.assertFalse(score('Expression: '+wrong,q)['correct'])
        q=make('second_moment',1,p,'probe',0)
        self.assertFalse(score('Expression: 23',q)['correct'])
        self.assertFalse(score('Expression: 17**2',q)['correct'])
        q=make('conditional_wait',3,p,'probe',0)
        self.assertFalse(score('Expression: (40-330/60)/2',q)['correct'])
        self.assertFalse(score('Expression: (40+330)/2',q)['correct'])

    def test_reassociation_and_scalar_shortcut(self):
        q=make('poisson_variance',3,[17,4,3,7,23,40,5],'probe',0)
        self.assertTrue(score('Expression: 270*17/60',q)['correct'])
        self.assertFalse(score('Expression: '+q['answer'],q)['correct'])

if __name__=='__main__':unittest.main()
