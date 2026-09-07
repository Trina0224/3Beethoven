import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from stats_teacher_compact import payload, judge, call, messages, summary, DATA, DOCS, read_json, intermediates, append


class CompactTests(unittest.TestCase):
    def test_complete_numeric_reference_and_wrong_steps(self):
        for q in read_json(DATA)['questions']:
            obj={'intermediates':{k:str(v) for k,v in intermediates(q).items()},
                 'answers':[{'question_id':q['id'],'expression':q['expression']}]}
            raw=json.dumps(obj)
            self.assertTrue(judge(raw,q)['accepted'])
            self.assertTrue(judge('```json\n'+raw+'\n```',q)['accepted'])
            self.assertFalse(judge(raw,q,'length')['accepted'])
            obj['intermediates'][next(iter(obj['intermediates']))]='-999'
            self.assertFalse(judge(json.dumps(obj),q)['accepted'])
            user=json.loads(messages(q)[1]['content'])
            self.assertNotIn('answer',user)
            self.assertNotIn('bindings',user)
            self.assertNotIn('semantics',user)

    def test_ambiguous_duplicate_and_truncated_envelopes_rejected(self):
        for raw in ('{"intermediates":{},"answers":[],"answers":[]}',
                    '```json\n{}\n```\n```json\n{}\n```',
                    '{}\n```json\n{}\n```', '```json\n{"answers":',
                    '{"intermediates":{},"answers":[]} trailing'):
            with self.assertRaises(ValueError):
                payload(raw)

    def test_common_contract_recovers_only_twelve_complete_old_B_answers(self):
        qs={q['id']:q for q in read_json(DATA)['questions']}
        results=[]
        for r in read_json(DOCS/'STATS_TEACHER_AB_EVIDENCE.json')['responses']:
            if 'intermediates object' in r['request']['messages'][0]['content']:
                q=qs[json.loads(r['request']['messages'][1]['content'])['question_id']]
                results.append(judge(r['text'],q,r['finish_reason']))
        self.assertEqual(sum(r['accepted'] for r in results),12)
        self.assertEqual(sum(r['classification']=='truncated_or_non_stop' for r in results),11)
        self.assertFalse(any(r['classification']=='equivalence_pending' for r in results))

    def test_final_number_only_and_wrong_formula_do_not_pass(self):
        q=next(q for q in read_json(DATA)['questions'] if q['development_family']=='affine_poisson')
        for expression in (q['answer'],'1+2'):
            raw=json.dumps({'intermediates':{k:str(v) for k,v in intermediates(q).items()},
                            'answers':[{'question_id':q['id'],'expression':expression}]})
            self.assertFalse(judge(raw,q)['accepted'])

    def test_spending_block_precedes_network(self):
        histories=[[dict(event='started',tag='lost')],
                   [dict(event='started',tag='x'),dict(event='response',tag='x',usage={})],
                   [dict(event='started',tag='x'),dict(event='response',tag='x',usage={'cost':0.49})]]
        for history in histories:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                for row in history:
                    append(root/'api_ledger.jsonl',row)
                fake=Mock()
                with patch.dict('sys.modules',{'requests':fake}):
                    with self.assertRaises(RuntimeError):
                        call(root,'unused','new',[{'role':'user','content':'test'}])
                    fake.post.assert_not_called()

    def test_family_failure_cannot_hide_under_total(self):
        rows=[dict(id=q['id'],family=q['development_family'],accepted=True,strict_json=True,classification='accepted') for q in read_json(DATA)['questions']]
        self.assertTrue(summary(rows,True)['passed'])
        self.assertFalse(summary(rows,False)['passed'])
        for row in rows[:2]:
            row.update(accepted=False,classification='mathematical_error')
        self.assertEqual(summary(rows,True)['accepted'],22)
        self.assertFalse(summary(rows,True)['passed'])


if __name__=='__main__':
    unittest.main()
