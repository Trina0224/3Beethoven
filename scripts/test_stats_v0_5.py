import tempfile
import unittest
from pathlib import Path
from fractions import Fraction as F
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from stats_curriculum_v0_5 import build,audit,item,TOPICS
from stats_holdout_v1 import questions
from stats_holdout_v2 import questions as q2
from stats_v0_3_common import make_curriculum
from generate_stats_v0_5 import ReservedClient,valid_target,parse_output


class ExpandedTests(unittest.TestCase):
    def test_families_and_no_duplicate_or_equivalent_choices(self):
        report=audit(questions()+q2()+make_curriculum())
        self.assertEqual([report[k]['families'] for k in ('train','validation','test')],[18,6,6])
        self.assertEqual([report[k]['n'] for k in ('train','validation','test')],[180,24,36])

    def test_independent_boundary_answers_for_all_30_families(self):
        expected={
            'poisson':[6,9,45,3,66],
            'expectation':[5,9,14,8,24],
            'uniform':[7,12,F(1,4),F(49,3),8],
            'type_i':[2,F(1,50),F(81,100),F(8,25),F(7,250)],
            'type_ii':[F(2,5),4,F(624,625),F(17,25),F(13,125)],
            'confidence':[8,4,12,2,22]}
        for topic in TOPICS:
            for family in range(5):
                self.assertEqual(F(item(topic,family,0)[1]),expected[topic][family],(topic,family))

    def test_target_schema_and_reference_gate(self):
        r=build()['train'][0]
        good=dict(answer_letter=r['answer_letter'],explanation='x'*60,common_mistake='Incorrect: '+'x'*20)
        self.assertTrue(valid_target(good,r))
        self.assertFalse(valid_target(dict(good,answer_letter='INVALID'),r))
        self.assertFalse(valid_target(dict(good,explanation='x'*1201),r))

    def test_recover_teacher_output_without_new_generation(self):
        self.assertEqual(parse_output('Answer: B\n{"answer_letter":"B","explanation":"kept","common_mistake":"kept"}')['explanation'],'kept')
        obj=parse_output('Answer: B\nOriginal explanation.\nCommon mistake: Incorrectly changing the value.')
        self.assertEqual(obj['explanation'],'Original explanation.')
        self.assertEqual(obj['common_mistake'],'Incorrectly changing the value.')
        with self.assertRaises(ValueError): parse_output('Answer: A\n{"answer_letter":"B"}')
        corrected=parse_output('{"answer_letter":"B is incorrect, the correct answer is A","explanation":"Unchanged."}')
        self.assertEqual(corrected['answer_letter'],'A')
        self.assertEqual(corrected['explanation'],'Unchanged.')

    def test_parallel_reservations_cannot_exceed_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            client=ReservedClient(Path(directory),'unused')
            client.reserved=596
            def call(i):
                try: client.call(str(i),[]); return True
                except RuntimeError: return False
            with patch('flight_run_stats_v0_3.TeacherClient.call',return_value='{}'):
                with ThreadPoolExecutor(max_workers=4) as pool:
                    accepted=list(pool.map(call,range(20)))
            self.assertEqual(sum(accepted),4)
            self.assertEqual(client.reserved,600)


if __name__=='__main__': unittest.main()
