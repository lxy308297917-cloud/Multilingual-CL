from pathlib import Path
import os
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics
ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'work/sw_fullbudget_baselines_20261028/evaluation_inputs/knowledge_adapter'/os.environ['KNOWLEDGE_LANG']
def mcf(line,task_name=None):
 instruction='The following are multiple choice questions (with answers) about '+line['subject'].replace('_',' ')+'.\n\n'
 query=instruction+line['question']+'\n'+''.join(f'{k}. {v}\n' for k,v in zip('ABCD',line['choices']))+'Answer:'
 return Doc(task_name=task_name,query=query,choices=[' A',' B',' C',' D'],gold_index='ABCD'.index(line['answer']),instruction=instruction)
TASKS_TABLE=[LightevalTaskConfig(name=f'full_knowledge_{os.environ["KNOWLEDGE_LANG"]}:{p.name}',prompt_function=mcf,
 hf_repo=str(p),hf_subset='default',hf_avail_splits=('test','validation'),evaluation_splits=('test',),
 few_shots_split='validation',few_shots_select='sequential',generation_size=1,
 metrics=[Metrics.loglikelihood_acc],stop_sequence=['\n'],version=1) for p in sorted(DATA.iterdir()) if p.is_dir()]
