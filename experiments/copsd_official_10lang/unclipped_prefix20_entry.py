"""One-factor diagnostic: official supported jsd_token_clip=0 (None internally).

All packing, optimizer, teacher, LoRA, rollout and 100-step LR horizon unchanged.
Not labelled the original published clip0.05 configuration.
"""
import json
from pathlib import Path
import prefix20_entry as prefix

original_init=prefix.seeded.original_init


def init(self,*args,**kwargs):
    original_init(self,*args,**kwargs)
    assert self.jsd_token_clip is None and self.beta==0
    assert self.fixed_teacher and not self.use_thinking_machines_loss
    assert self.top_k_loss is None
    p=Path(self.args.output_dir);p.mkdir(parents=True,exist_ok=True)
    (p/'ABLATION_ACTUAL.json').write_text(json.dumps({
        'protocol':'project_SW_component_clip_one_factor_v1',
        'published_reference_component_clip':.05,'actual_component_clip':None,
        'core_loss_replaced':False,'official_supported_flag':'--jsd_token_clip 0',
        'same_100_update_scheduler_horizon_stop_after20':True,
        'scope':'Single-factor diagnostic, not the released clip0.05 reproduction or proof of efficacy.'},indent=2)+'\n')


if __name__=='__main__':
    prefix.seeded.original_init=init
    prefix.main()
