import datetime,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import budget
class BudgetTests(unittest.TestCase):
 def test_all_parallel_queues_charged_once_and_unrelated_queue_excluded(self):
  with tempfile.TemporaryDirectory() as temp:
   queues=Path(temp)
   for name in ('paired-cap-smoke-20261021','paired-cap-A-20261021','paired-cap-R-20261021','other-experiment'):
    run=queues/name/'runs/id';run.mkdir(parents=True)
    (run/'job.json').write_text(json.dumps({'script':str(budget.ROOT/'run_parallel.sh'),'started_at':'2026-10-21 20:00:00','finished_at':'2026-10-21 20:01:00','gpu_uuids':['GPU-test'],'phase':'completed'}))
   with patch.object(budget,'QUEUES',queues):got=budget.ledger(datetime.datetime(2026,10,21,20,2))
   self.assertEqual(len(got['jobs']),3);self.assertAlmostEqual(got['physical_gpu_hours_including_failures'],.05)
if __name__=='__main__':unittest.main()
