"""BN matched component-clipping diagnostic; official loss is unchanged."""
import json
import os
from pathlib import Path
import prefix20_entry as prefix

original_init = prefix.seeded.original_init

def init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    expected = .05 if os.environ['COPSD_BN_CLIP_MODE'] == 'clip' else None
    assert self.beta == 0 and self.jsd_token_clip == expected
    assert self.fixed_teacher and not self.use_thinking_machines_loss
    assert self.top_k_loss is None
    p = Path(self.args.output_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p/'ABLATION_ACTUAL.json').write_text(json.dumps({
        'protocol': 'project_BN_component_clip_matched20_v1',
        'actual_component_clip': self.jsd_token_clip,
        'beta': self.beta, 'core_loss_replaced': False,
        'scheduler_horizon': 100, 'stop_after_updates': 20,
        'scope': 'BN one-factor independent-development diagnosis, not official full100 reproduction or established repair.'
    }, indent=2)+'\n')

if __name__ == '__main__':
    prefix.seeded.original_init = init
    prefix.main()
