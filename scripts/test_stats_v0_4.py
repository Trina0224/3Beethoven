import unittest
from collections import Counter
from stats_v0_3_common import make_curriculum,group_split
from stats_holdout_v1 import questions as old_questions
from stats_holdout_v2 import questions,validate
from run_stats_v0_4 import examples,metrics


class RepairTests(unittest.TestCase):
    def test_teacher_provenance_and_group_separation(self):
        records=[dict(r,explanation='Original explanation with option A mentioned.',common_mistake='Original mistake text.') for r in make_curriculum()]
        train,val=group_split(records)
        a,b=examples(train),examples(val)
        self.assertEqual((len(a),len(b)),(240,60))
        self.assertFalse({r['source_id'] for r in a}&{r['source_id'] for r in b})
        for r in records:
            group=examples([r])
            self.assertEqual(Counter(x['target'] for x in group[:4]),dict.fromkeys('ABCD',1))
            self.assertEqual(group[4]['target'],f"Answer: {r['answer_letter']}\n\nExplanation: {r['explanation']}\n\nCommon misconception: {r['common_mistake']}")

    def test_new_probe_is_separate_and_balanced(self):
        self.assertEqual(validate(make_curriculum()+old_questions())['n'],24)
        rows=questions()
        # Independent hand-computed boundary examples for all six families.
        expected={0:'20',3:'52',4:'8',7:'68',8:'12',11:'192',
                  12:'3*0.10*0.90^2',15:'6*0.10*0.90^5',
                  16:'1-(1/10)^2',19:'1-(2/5)^2',20:'4',23:'10'}
        for i,answer in expected.items():
            self.assertEqual(rows[i]['choices']['ABCD'.index(rows[i]['answer_letter'])],answer)

    def test_metrics_reject_incomplete_and_measure_content(self):
        qs=questions()[:1]
        rows=[dict(id=qs[0]['id'],category=qs[0]['category'],shift=s,expected='ABCD'[s],
                   predicted='ABCD'[s],correct=True,original_choice_index=0,hit_token_limit=False) for s in range(4)]
        self.assertEqual(metrics(rows,qs)['all_four_correct'],1)
        self.assertEqual(metrics(rows,qs)['semantically_consistent'],1)
        with self.assertRaises(AssertionError): metrics(rows[:3],qs)


if __name__=='__main__': unittest.main()
