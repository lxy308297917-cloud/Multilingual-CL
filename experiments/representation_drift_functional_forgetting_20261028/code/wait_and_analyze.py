"""CPU-only postprocessing after the four genuine matched extraction receipts exist."""
from pathlib import Path
import sys,os,time,subprocess,traceback,json
ROOT=Path(__file__).resolve().parents[3];H=Path('/data0/users/lixiaoyang')
REPORT=ROOT/'docs/results/representation_drift_functional_forgetting_20261028';WORK=ROOT/'work/representation_drift_functional_forgetting_20261028'
CODE=Path(__file__).parent;PY=H/'ssu-migration/envs/cl/bin/python'
sys.path.insert(0,str(ROOT/'outputs/sw_fullbudget_baselines_20261028/code'))
from settings import read,write,sha
try:
 protocol=read(REPORT/'PROTOCOL_FROZEN.json')
 while not all((WORK/model/'first/DONE.json').exists() for model in protocol['models']):
  if (WORK/'QUEUE_FAILED.json').exists():raise RuntimeError('Extraction failed; inspect recovery before analysis')
  time.sleep(30)
 env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
 started=time.time()
 for model in protocol['models']:
  receipt=WORK/model/'first/CONTENT_V2_ACCEPTANCE.json'
  if not receipt.exists():
   subprocess.run([str(PY),str(CODE/'rescore_math_v2.py'),'--model',model],cwd=ROOT,env=env,check=True)
  value=read(receipt);assert value['parser_sha256']==read(REPORT/'MATH_CONTENT_V2_PROTOCOL.json')['parser_sha256']
  assert value['content_prediction_sha256']==sha(receipt.parent/'PREDICTIONS_CONTENT_V2.jsonl')
 # Reuse the independent complete-training acceptance; no model updates or new evaluation.
 for method in ('Dense','Element','Protected'):
  subprocess.run([str(PY),str(ROOT/'outputs/sw_fullbudget_baselines_20261028/closeout/audit_training.py'),'--method',method],cwd=ROOT,env=env,check=True)
 subprocess.run([str(PY),str(CODE/'analyze.py')],cwd=ROOT,env=env,check=True)
 subprocess.run([str(PY),str(CODE/'plot.py')],cwd=ROOT,env=env,check=True)
 costs=[];total=0.;unclosed=[]
 for start in WORK.glob('GPU_START_*.json'):
  a=read(start);end=start.with_name(start.name.replace('GPU_START_','GPU_END_',1))
  if not end.exists():unclosed.append(str(start));continue
  b=read(end);seconds=b['finished_unix']-a['started_unix'];assert seconds>=0;total+=seconds
  costs.append(dict(start=str(start),end=str(end),gpu_seconds=seconds,**b))
 write(REPORT/'COSTS.json',dict(contexts=costs,known_GPU_hours=total/3600,unclosed_contexts=unclosed,
  CPU_analysis_wall_seconds=time.time()-started,no_time_stop_threshold=True,original_baseline_training_cost_excluded=True,
  missing_exit_times_not_imputed=True))
 training={m:read(ROOT/'outputs/sw_fullbudget_baselines_20261028/training_acceptance'/f'{m}.json') for m in ('Dense','Element','Protected')}
 write(REPORT/'PARAMETER_UPDATE_CONTROLS.json',dict(models={m:dict(final_net_displacement=v['final_net_displacement'],cumulative_actual_path=v['cumulative_actual_path']) for m,v in training.items()},
  caveat='Only three adapted endpoints; update norm constant within a model. Descriptive cross-model comparison, no independent-sample or causal norm attribution.'))
 analysis=read(REPORT/'DRIFT_FORGETTING_ANALYSIS.json')
 write(WORK/'STAGE1_OUTPUTS_READY.json',dict(finished_unix=time.time(),models=protocol['models'],
  patching_gate_passed=analysis['patching_gate_passed'],
  manual_figure_review_and_scientific_interpretation_pending=True,
  patching_if_gate_passed_not_yet_executed=True,full_task_complete=False))
except BaseException as error:
 write(WORK/'ANALYSIS_FAILED.json',dict(error=str(error),traceback=traceback.format_exc(),unix=time.time()));raise
