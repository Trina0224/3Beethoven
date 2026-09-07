import json
import unittest
from prepare_stats_first_batch import parse, ingest
from stats_consolidation_pilot import build


class FirstBatchTests(unittest.TestCase):
    def test_duplicate_ids_rejected(self):
        a={'question_id':'q','expression':'1+2'}
        with self.assertRaises(ValueError):
            parse(json.dumps({'answers':[a,a]}),{'q'})

    def test_fence_does_not_change_expression(self):
        raw='```json\n'+json.dumps({'answers':[{'question_id':'q','expression':'2*(3/60)'}]})+'\n```'
        self.assertEqual(parse(raw,{'q'})[0][0]['expression'],'2*(3/60)')

    def test_wrong_interval_and_truncation_rejected(self):
        q=next(q for s in build()[0] for q in s['questions'] if q['category']=='interval')
        r={'accepted':{},'attempts':[]}
        raw=json.dumps({'answers':[{'question_id':q['id'],'expression':q['expression']}]})
        ingest(r,raw,{q['id']:q},0,'length')
        self.assertFalse(r['accepted'])
        wrong=json.dumps({'answers':[{'question_id':q['id'],'expression':'('+q['expression']+')*2'}]})
        ingest(r,wrong,{q['id']:q},1)
        self.assertFalse(r['accepted'])
        ingest(r,raw,{q['id']:q},2)
        self.assertIn(q['id'],r['accepted'])


if __name__=='__main__':
    unittest.main()
