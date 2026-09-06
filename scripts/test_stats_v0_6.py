import unittest
from fractions import Fraction as F
from stats_holdout_v0_6 import build,audit
from run_stats_rotation_v1 import rotate
from stats_curriculum_v0_5 import build as previous

class FrozenStoryTest(unittest.TestCase):
    def test_pairing_preserves_labels_sources_and_validation(self):
        import json
        from pathlib import Path
        from prepare_stats_v0_6_pairs import assemble
        from run_stats_v0_6 import examples
        from run_stats_v0_5 import examples as old_examples
        old=json.loads((Path(__file__).resolve().parent.parent/'docs'/'STATS_V0_5_TEACHER_DATA.json').read_text())
        paired=assemble(old); sources={r['id']:r for r in old['train']}
        self.assertEqual(len(paired),180)
        for r in paired:
            source=sources[r['contrast_id']]
            self.assertEqual(r['contrast_explanation'],source['explanation'])
            self.assertEqual(r['category'],source['category'])
            self.assertNotEqual(r['family'],source['family'])
        new=examples(paired); prior=old_examples(old['train'])
        self.assertEqual(new[::2],prior[::2])
        self.assertEqual(examples(old['validation']),old_examples(old['validation']))
        self.assertIn('\nFor comparison,',new[1]['prompt'])
        self.assertIn('\n\nContrasting calculation:',new[1]['target'])
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
