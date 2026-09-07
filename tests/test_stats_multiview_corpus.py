import hashlib
import re
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from stats_multiview_corpus import read, replace_numbers, task_key, validate_question, SUFFIX, ROOT

class CorpusContract(unittest.TestCase):
    def test_targets_and_unseen_prompts(self):
        c=read(ROOT/'docs/STATS_MULTIVIEW_CORPUS.json')
        self.assertEqual(len(c['train']),516)
        for r in c['train']:
            self.assertEqual(hashlib.sha256(r['target'].encode()).hexdigest(),r['target_sha256'])
        self.assertFalse({r['prompt'] for r in c['train']} &
            {q['question']+SUFFIX for q in c['transfer_probe'] if q['wording']=='unseen'})

    def test_crossed_probe_isolates_numeric_axis(self):
        probe=read(ROOT/'docs/STATS_MULTIVIEW_CORPUS.json')['transfer_probe']
        self.assertEqual(len(probe),48)
        for i in range(0,48,4):
            a,b,c,d=probe[i:i+4]
            self.assertEqual(task_key(a),task_key(b))
            self.assertEqual(task_key(c),task_key(d))
            self.assertNotEqual(task_key(a),task_key(c))
            self.assertEqual(replace_numbers(a['question'],a,a),a['question'])
            self.assertEqual(re.sub(r'\d+(?:\.\d+)?','N',a['question']),
                             re.sub(r'\d+(?:\.\d+)?','N',c['question']))
            for q in (a,b,c,d):validate_question(q)

if __name__=='__main__':unittest.main()
