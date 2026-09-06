import json
import unittest
from pathlib import Path
from check_teacher_perturbations_v0_14 import build,digest
from formulation_grader import grade
from prepare_verified_distillation_v0_14 import build as targets


class TeacherProtocolTests(unittest.TestCase):
    def test_frozen_pairs_and_student_separation(self):
        repo=Path(__file__).resolve().parents[1]
        data=build()
        self.assertEqual(digest(data),'14bc722c03b772df8750dbb38c69e900b86fe88308a08f806933b4245b2f5852')
        student=json.loads((repo/'docs/STATS_V0_14_FROZEN_QUESTIONS.json').read_text())
        identities={tuple(q['identity']) for rows in student.values() for q in rows}
        for pair in data['pairs']:
            a,b=pair['questions']
            self.assertEqual(a['answer']==b['answer'],pair['group'] in ('wording','unit'))
            for q in (a,b):
                self.assertNotIn(tuple(q['identity']),identities)
                self.assertTrue(grade(q['expression'],q)['math_correct'])

    def test_saved_teacher_targets_reproduce(self):
        repo=Path(__file__).resolve().parents[1]
        self.assertEqual(targets(repo),json.loads((repo/'docs/STATS_V0_14_VERIFIED_DISTILLATION.json').read_text()))


if __name__=='__main__':unittest.main()
