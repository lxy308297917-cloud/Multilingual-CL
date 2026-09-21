"""Stateless balanced calibration windows derived from optimizer step."""
import json
from pathlib import Path
from safe_state import file_sha

class CalibrationSchedule:
 def __init__(self,path,sha256,per_task=8,refresh_steps=100):
  if per_task<1 or refresh_steps<1:raise ValueError('Invalid schedule')
  if file_sha(path)!=sha256:raise ValueError('Calibration identity mismatch')
  self.pools={};self.per_task=per_task;self.refresh_steps=refresh_steps
  for line in Path(path).open():
   row=json.loads(line)
   if row['split']=='scoring':self.pools.setdefault(row['task'],[]).append(row)
  if set(self.pools)!={'instruction','reading','math'}:raise ValueError('Task identity mismatch')
  if any(len(rows)!=256 for rows in self.pools.values()):raise ValueError('Expected frozen 256 scoring rows per task')
  if per_task>256:raise ValueError('Cannot repeat a sample within one window')
 def window(self,optimizer_step):
  if type(optimizer_step)!=int or optimizer_step<0:raise ValueError('Invalid optimizer step')
  # A forced refresh after zero-LR warmup uses the same initial window.
  offset=(optimizer_step//self.refresh_steps*self.per_task)%256
  rows={task:[pool[(offset+i)%256] for i in range(self.per_task)] for task,pool in self.pools.items()}
  return rows
