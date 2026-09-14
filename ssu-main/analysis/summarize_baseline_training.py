"""Summarize measured training scope and saved-weight audits; no capability claims."""
import csv,io,json,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];S=R.parent/'experiments/ig_baseline_suite_v1'
def generate():
 c=json.loads((R/'configs/ig_baseline_train_v1.json').read_text());total=1543714304;rows=[]
 for name,setting in c['methods'].items():
  p=Path(c['output_root'])/name
  def load(rel):return json.loads((p/rel).read_text()) if (p/rel).exists() else {}
  info=load('EXPERIMENT_INFO.json');done=load('TRAINING_DONE.json');mask=load('selection/MASK_DONE.json');audit=load('weight_update_audit.json');run=load('RUNNING.json')
  live=any((Path('/proc')/str(run.get(k,-1))/'cmdline').exists() for k in ['pid','child_pid'])
  state='failed' if (p/'FAILED.json').exists() else 'complete' if done else 'running' if live else 'not_started' if not info else 'needs_inspection'
  eligible=mask.get('gradient_eligible_elements')
  if name=='full_fft':eligible=total
  elif name=='layers1_5_23_27_fft':eligible=467978240
  note='静态梯度允许范围；实际发生数值变化另列。'
  if name=='mofo15':eligible=None;note='每步动量阈值名义15%，并列可超过15%；静态mask不能证明逐步更新比例。'
  rows.append({'method':name,'state':state,'budget_steps':c['max_steps'],'completed_steps':done.get('steps'),'unique_base_parameters':total,'gradient_eligible_elements':eligible,'gradient_eligible_percent':None if eligible is None else eligible/total*100,'actually_changed_elements':audit.get('changed_elements',audit.get('selected_changed_elements')),'frozen_changed_elements':audit.get('frozen_changed_elements'),'weight_audit_passed':audit.get('passed'),'wall_minutes_including_pause_and_save':None if not done or not info else (done['completed_unix']-info['started_unix'])/60,'extra_calibration_steps':setting.get('calibration_steps',0),'note':note,'source':str(p)})
 out=io.StringIO();w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);(S/'training_parameter_cost_audit.csv').write_text(out.getvalue())
 lines=['# 训练参数与代价审计','','所有方法共享Base、Igbo train10k、seed42、15000 steps、batch2、长度512、学习率5e-5、weight decay0.01。只保存最终模型；中断后无optimizer断点可用。','','|方法|状态|梯度允许范围%|最终数值变化参数数|冻结位置变化|权重验收|','|---|---|---:|---:|---:|---|']
 for x in rows:
  fmt=lambda v:'—' if v is None else str(v)
  ratio='动态名义15%' if x['method']=='mofo15' else '—' if x['gradient_eligible_percent'] is None else f"{x['gradient_eligible_percent']:.4f}"
  lines.append('|'+ '|'.join([x['method'],x['state'],ratio,fmt(x['actually_changed_elements']),fmt(x['frozen_changed_elements']),fmt(x['weight_audit_passed'])])+'|')
 lines+=['','允许梯度更新的参数范围与最终数值变化参数数不同，BF16及小更新可能使允许范围中的权重保持同值。共享embedding/head按唯一参数计数，不能采用旧日志中重复计数的分母。','SSU使用101个实际可用English校准blocks（请求500）；Top20使用128个Igbo blocks；LoTA有额外100步校准并在选择后恢复Base。各方法选择成本单列，不能称绝对等计算预算。','全部训练与下游能力结果分开；mask或权重审计通过不证明下游效果。时间字段含人工暂停和保存等待，不用于GPU纯计算效率比较。']
 (S/'TRAINING_PARAMETER_COST_AUDIT_ZH.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':generate()
