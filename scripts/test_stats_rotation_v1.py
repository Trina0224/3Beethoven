import unittest
from run_stats_rotation_v1 import rotate,original_index,metrics
from stats_holdout_v1 import questions

class RotationTests(unittest.TestCase):
    def test_rotations_preserve_semantic_gold_and_cover_positions(self):
        for r in questions():
            correct=r["choices"]["ABCD".index(r["answer_letter"])]
            variants=[rotate(r,s) for s in range(4)]
            self.assertEqual({v["answer_letter"] for v in variants},set("ABCD"))
            for shift,v in enumerate(variants):
                self.assertEqual(v["choices"]["ABCD".index(v["answer_letter"])],correct)
                self.assertEqual(original_index(v["answer_letter"],shift),"ABCD".index(r["answer_letter"]))
            self.assertEqual(variants[0],r)

    def test_position_fixation_is_not_semantic_consistency(self):
        self.assertEqual([original_index("A",s) for s in range(4)],[0,1,2,3])
        self.assertIsNone(original_index("INVALID",0))
        result=[]
        for r in questions():
            for shift in range(4):
                v=rotate(r,shift)
                result.append({"id":r["id"],"category":r["category"],"shift":shift,"expected":v["answer_letter"],
                               "predicted":"A","correct":v["answer_letter"]=="A","raw":"A",
                               "original_choice_index":shift,"hit_token_limit":False})
        scored=metrics(result,{r["id"]:{"raw":"A"} for r in questions()})
        self.assertEqual(scored["overall"]["accuracy"],0.25)
        self.assertEqual(scored["semantically_consistent"],0)
        self.assertEqual(scored["all_four_correct"],0)
        self.assertEqual(scored["constant_letter_questions"],60)

if __name__=="__main__":
    unittest.main()
