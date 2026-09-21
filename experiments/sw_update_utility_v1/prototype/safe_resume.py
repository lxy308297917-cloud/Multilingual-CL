"""Single-process Trainer resume using only non-pickle state files."""
from pathlib import Path
import random
import numpy as np
import torch
from safe_state import save_state,load_state

class SafeResumeMixin:
 def _save_optimizer_and_scheduler(self,output_dir):
  if self.args.world_size!=1:raise ValueError('Safe resume currently supports one process only')
  folder=Path(output_dir)
  save_state(folder/'safe_optimizer',self.optimizer.state_dict())
  save_state(folder/'safe_scheduler',self.lr_scheduler.state_dict())
 def _save_rng_state(self,output_dir):
  rng={'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.random.get_rng_state()}
  if torch.cuda.is_available():rng['cuda']=torch.cuda.random.get_rng_state_all()
  save_state(Path(output_dir)/'safe_rng',rng)
 def _load_optimizer_and_scheduler(self,checkpoint):
  if checkpoint is None:return
  self.optimizer.load_state_dict(load_state(Path(checkpoint)/'safe_optimizer'))
  self.lr_scheduler.load_state_dict(load_state(Path(checkpoint)/'safe_scheduler'))
 def _load_rng_state(self,checkpoint):
  if checkpoint is None:return
  rng=load_state(Path(checkpoint)/'safe_rng')
  random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.random.set_rng_state(rng['cpu'])
  if torch.cuda.is_available():torch.cuda.random.set_rng_state_all(rng['cuda'])
