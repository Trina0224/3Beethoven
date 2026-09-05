import unittest
from pathlib import Path
from stats_holdout_v1 import questions,validate
from stats_v0_3_common import make_curriculum,read_frozen

class HoldoutTests(unittest.TestCase):
    def test_balance_and_overlap(self):
        old=make_curriculum()+read_frozen(Path(__file__).with_name("flight_run_stats_v0_2.py"))
        self.assertEqual(validate(old)["questions"],60)

    def test_numeric_gold_independent_calculation(self):
        expected={1:str(4*3),2:str(2+5),3:str(18//9),5:str(3*4-2*7+5),6:str((7-1)//2),
                  8:str(11-11),9:str((-6+10)//2),10:str(2*9-14),11:"They have equal means, but X has larger variance",
                  12:"4 minutes",14:"0.02",15:str(200*5//100),18:"0.15",19:str(100*20//100),
                  22:f"{2*2.576*3:.3f}",24:str(2*(18-10)),25:str(12//2),27:str(4+9),28:str(4*9),
                  31:str(7+2*2+1),32:str(2*5),34:"0.3",36:"5",37:"1/2",38:"1.5",39:str((8-2)**2//12),
                  40:"sqrt(12)",41:"4/3",42:str(1+5),44:"0.04",46:"0.05",47:"0",50:"0",54:str(80*75//100),
                  55:str(3**2),56:str(1000//10),60:str(2//2)}
        rows=questions()
        for index,value in expected.items():
            r=rows[index-1]
            self.assertEqual(r["choices"]["ABCD".index(r["answer_letter"])],value,r["id"])

    def test_answers_not_in_prompt_metadata(self):
        from stats_v0_3_common import prompt_for
        for r in questions():
            text=prompt_for(r)
            self.assertNotIn(r["reference_reason"],text)
            self.assertNotIn("reference_reason",text)

if __name__=="__main__":
    unittest.main()
