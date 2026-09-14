"""Exercise actual full-suite command construction without running inference."""
import sys,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
import baseline_eval_worker as worker
class Captured(Exception):pass
class FullPlan(unittest.TestCase):
 def test_full_sample_limits_and_all_knowledge_subjects(self):
  c=json.loads((R/'configs/ig_baseline_eval_v1.json').read_text())
  for job in c['tasks']:
   with self.subTest(task=job['name']),tempfile.TemporaryDirectory() as temp:
    command=[]
    def capture():command.extend(sys.argv);raise Captured()
    with patch('lighteval.__main__.app',capture):
     with self.assertRaises(Captured):worker.light(c,job,Path(c['base_model']),Path(temp),False)
    self.assertIn('batch_size=1',command[2]);self.assertEqual(command[5],str(R/'evaluation/custom_multilingual_tasks.py'))
    if job['generative']:self.assertEqual(command[command.index('--max-samples')+1],'500')
    else:self.assertNotIn('--max-samples',command)
    if '{subject}' in job['task']:self.assertEqual(len(command[3].split(',')),57)
if __name__=='__main__':unittest.main()
