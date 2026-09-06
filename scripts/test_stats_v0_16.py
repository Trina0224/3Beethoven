import unittest
from collections import Counter
from stats_curriculum_v0_16 import build,digest,prompt,score

class CurriculumTests(unittest.TestCase):
    def test_frozen_split_and_contrasts(self):
        d=build();self.assertEqual(digest(d),'db2a2a3513e680ca64cea2d2a80e84f23e332377b140bcbc5150ccd0311c0f6c')
        ids={s:{tuple(q['identity']) for q in rows} for s,rows in d.items()}
        for a,b in (('train','validation'),('train','test'),('validation','test')):self.assertFalse(ids[a]&ids[b])
        self.assertEqual(set(Counter(q['story_id'] for q in d['train']).values()),{4})
        for rows in d.values():
            for q in rows:
                self.assertTrue(score('Expression: '+q['expression'],q)['correct'])
                self.assertFalse(score('Expression: X',q)['correct'])
                self.assertNotIn('Var(Y)=',prompt(q))

    def test_confused_quantities_fail(self):
        d=build();q=next(q for q in d['train'] if q['task']=='variance')
        b=q['bindings']
        self.assertFalse(score('Expression: '+b['scale']+'*'+b['variance'],q)['correct'])
        self.assertFalse(score('Expression: '+q['expression']+'+'+b['offset'],q)['correct'])
        self.assertNotIn('E[Y**2]',prompt(q,'full'))

    def test_teacher_review_does_not_change_student_grader(self):
        from review_teacher_v0_16 import teacher_score
        q=next(q for q in build()['train'] if q['task']=='variance')
        a=q['bindings']['scale'];partial=q['expression'].replace(a+'**2',str(int(a)**2),1)
        self.assertTrue(teacher_score('Expression: '+partial,q)['correct'])
        self.assertFalse(score('Expression: '+partial,q)['correct'])
        self.assertFalse(teacher_score('Expression: '+partial+'+'+q['bindings']['offset'],q)['correct'])

if __name__=='__main__':unittest.main()
