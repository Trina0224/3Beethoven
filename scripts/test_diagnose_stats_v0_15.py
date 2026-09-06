import unittest
from diagnose_stats_v0_15 import build
from formulation_grader import grade

class DiagnosticTests(unittest.TestCase):
    def test_reference_and_no_gold_binding_repair(self):
        for q in build():
            self.assertTrue(grade('Expression: '+q['expression'],q)['math_correct'])
            self.assertFalse(grade('Expression: 0',q)['math_correct'])
            self.assertIsNone(grade('Expression: unknown',q)['math_correct'])

    def test_scalar_extraction_does_not_relax_formulation(self):
        rows=build()
        extraction=next(q for q in rows if q['stage']=='extract_mean')
        self.assertTrue(grade(extraction['answer'],extraction)['math_correct'])
        formula=next(q for q in rows if q['stage']=='original')
        self.assertIsNone(grade(formula['answer'],formula)['math_correct'])

    def test_paired_independent_conditions(self):
        rows=build()
        self.assertEqual(len(rows),128)
        self.assertEqual(len({q['id'] for q in rows}),128)
        self.assertEqual(len({q['story_id'] for q in rows}),16)
        for q in rows:
            if q['stage']!='hinted':
                self.assertNotIn('Var(a*X+b)=',q['prompt'])

if __name__=='__main__':unittest.main()
