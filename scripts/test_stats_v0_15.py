import unittest
from stats_curriculum_v0_15 import build,digest,score


class CurriculumTests(unittest.TestCase):
    def test_frozen_references_and_split_separation(self):
        data=build()
        self.assertEqual(digest(data),'d7df943fa318ffbaeffabd12834a8f345c54f71c48f8c8cc1cbef8c78ba55f1f')
        ids=[{tuple(q['identity'][:len(q['identity'])-2] if 'stage' in q['identity'] else q['identity']) for q in rows} for rows in data.values()]
        self.assertFalse(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
        for rows in data.values():
            for q in rows:self.assertTrue(score(q['expression'],q)['correct'])

    def test_wrong_quantity_still_fails(self):
        q=next(q for q in build()['test'] if q['category']=='moment')
        b=q['bindings']
        self.assertFalse(score(f"{b['scale']}**2*{b['variance']}",q)['correct'])


if __name__=='__main__':unittest.main()
