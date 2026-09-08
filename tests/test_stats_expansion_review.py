"""Meaningful rejection boundaries for post-output teacher review."""
import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from stats_expansion_teacher_review import review,event_proof
from stats_expansion_teacher_format import intermediates
ROOT=Path(__file__).resolve().parents[1]

class ReviewBoundaries(unittest.TestCase):
    def setUp(self):
        self.q=json.loads((ROOT/'docs/STATS_EXPANSION_QUESTIONS.json').read_text())['train'][0]
        self.obj={'intermediates':{k:str(v) for k,v in intermediates(self.q).items()},'answers':[{'question_id':self.q['id'],'expression':self.q['expression']}]}
    def judge(self,text):return review({'text':text,'finish_reason':'stop'},self.q)
    def test_unique_echo_preserves_expression(self):
        raw=json.dumps({'question_id':self.q['id'],'question':self.q['question'],'output_structure':self.obj})
        r=self.judge(raw);self.assertTrue(r['accepted']);self.assertEqual(r['expression'],self.q['expression'])
    def test_only_one_extra_closing_brace(self):
        self.assertTrue(self.judge(json.dumps(self.obj)+'}')['accepted'])
        self.assertFalse(self.judge(json.dumps(self.obj)+'}}')['accepted'])
    def test_wrong_echo_rejected(self):
        self.assertFalse(self.judge(json.dumps({'question_id':self.q['id'],'question':'Different problem','output_structure':self.obj}))['accepted'])
    def test_competing_envelopes_rejected(self):
        obj=dict(self.obj,output_structure=self.obj)
        self.assertFalse(self.judge(json.dumps(obj))['accepted'])
    def test_duplicate_json_key_rejected(self):
        raw=json.dumps(self.obj).replace('"intermediates":','"intermediates":{},"intermediates":',1)
        self.assertFalse(self.judge(raw)['accepted'])
    def test_wrong_intermediate_rejected(self):
        self.obj['intermediates']['p_detect_a']='0'
        self.assertFalse(self.judge(json.dumps(self.obj))['accepted'])
    def test_numeric_equality_not_symbolic_proof(self):
        with self.assertRaises((ValueError,AssertionError)):event_proof(self.q['answer'],self.q)
    def test_wrong_event_rejected(self):
        p,z=self.q['semantics']['p'],self.q['semantics']['p_b']
        with self.assertRaises((ValueError,AssertionError)):event_proof(f'({p})*({z})',self.q)
if __name__=='__main__':unittest.main()
