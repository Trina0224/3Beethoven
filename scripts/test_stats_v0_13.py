import unittest
from stats_curriculum_v0_13 import build, canonical, make, score


class FormulationTests(unittest.TestCase):
    def test_frozen_generation(self):
        data=build()
        self.assertEqual(data,build())
        self.assertEqual(len(data['test']),96)

    def test_unit_conversion(self):
        q=make('poisson_time',(31,90,2,3),'test',0)
        self.assertTrue(score(q['target'],q)['correct'])
        self.assertFalse(score(q['target'].replace('31*(90/60)','31*90'),q)['correct'])

    def test_wrong_binding(self):
        q=make('poisson_time',(31,90,2,3),'test',0)
        self.assertFalse(score(q['target'].replace('duration_minutes=90/60','duration_minutes=90'),q)['correct'])

    def test_answer_only_needs_review(self):
        q=make('poisson_time',(31,90,2,3),'test',0)
        judged=score(q['target'].replace(q['expression'],q['answer']),q)
        self.assertFalse(judged['correct'])
        self.assertTrue(judged['review_required'])

    def test_event_contrast(self):
        q=make('exactly_one',(31,90,2,3),'test',0)
        other=make('at_least_one',(31,90,2,3),'test',0)
        self.assertFalse(score(other['target'],q)['correct'])

    def test_commutative(self):
        self.assertEqual(canonical('3*(90/60)'),canonical('(90/60)*3'))

    def test_unsafe_and_duplicate(self):
        q=make('poisson_time',(31,90,2,3),'test',0)
        self.assertTrue(score(q['target'].replace(q['expression'],'__import__("os")'),q)['invalid'])
        self.assertTrue(score(q['target'].replace('Bindings: ','Bindings: rate_per_minute=31; '),q)['invalid'])


if __name__=='__main__': unittest.main()
