import unittest
from formulation_grader import grade
from stats_curriculum_v0_14 import build


class FormulationTests(unittest.TestCase):
    def setUp(self):
        self.q={q['category']:q for q in build()['train']}

    def test_unlabelled_formula(self):
        for q in self.q.values():
            result=grade(q['expression'],q)
            self.assertTrue(result['math_correct'])
            self.assertFalse(result['format_exact'])

    def test_model_bindings(self):
        q=self.q['poisson_scaled'];b=q['bindings']
        raw=f"m={b['mean']}; a={b['scale']}\na**2*m"
        self.assertTrue(grade(raw,q)['math_correct'])

    def test_no_reference_binding_injection(self):
        self.assertIsNone(grade('scale**2*mean',self.q['poisson_scaled'])['math_correct'])

    def test_final_error_overrides_correct_setup(self):
        q=self.q['moment']
        self.assertFalse(grade(q['expression']+'\n0',q)['math_correct'])

    def test_answer_only_needs_review(self):
        q=self.q['moment']
        self.assertIsNone(grade(q['answer'],q)['math_correct'])

    def test_missing_mean_square_is_wrong(self):
        q=self.q['moment'];b=q['bindings']
        self.assertFalse(grade(f"{b['scale']}**2*{b['variance']}+{b['offset']}**2",q)['math_correct'])

    def test_unsafe_and_unknown_final_stay_pending(self):
        q=self.q['moment']
        for raw in ('__import__("os").system("false")',q['expression']+'\nunknown_name'):
            self.assertIsNone(grade(raw,q)['math_correct'])


if __name__=='__main__':unittest.main()
