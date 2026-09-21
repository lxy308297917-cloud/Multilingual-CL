"""Capture fully accumulated target loss gradients before Trainer clips them."""
class RawGradientCaptureMixin:
 def training_step(self,*args,**kwargs):
  loss=super().training_step(*args,**kwargs)
  opt=self.optimizer
  while not hasattr(opt,'engine') and hasattr(opt,'optimizer'):opt=opt.optimizer
  if (hasattr(opt,'engine') and self.accelerator.sync_gradients and
      opt.engine.settings['mode']=='utility' and opt.engine.needs_refresh()):
   opt.target_score_grads={n:p.grad.detach().cpu().clone() for n,p in opt.engine.params.items()}
   self.raw_gradient_capture_count=getattr(self,'raw_gradient_capture_count',0)+1
  return loss
