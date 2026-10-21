"""TE extension of the BN one-factor clip diagnostic; official loss unchanged."""
import json
import os
from pathlib import Path

import prefix20_entry as prefix

original_init = prefix.seeded.original_init


def init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    mode = os.environ['COPSD_TE_CLIP_MODE']
    assert mode in ['clip', 'unclip']
    expected = .05 if mode == 'clip' else None
    assert self.beta == 0 and self.jsd_token_clip == expected
    assert self.fixed_teacher and not self.use_thinking_machines_loss and self.top_k_loss is None
    assert self.args.seed == self.args.data_seed == 42
    p = Path(self.args.output_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / 'ABLATION_ACTUAL.json').write_text(json.dumps({
        'protocol': 'project_TE_component_clip_matched20_v1', 'language': 'te',
        'actual_component_clip': self.jsd_token_clip, 'beta': self.beta,
        'core_loss_replaced': False, 'scheduler_horizon': 100, 'stop_after_updates': 20,
        'scope': 'Fresh matched TE20 diagnostic. Not official100 reproduction, established repair, final language selection or sequential training.'
    }, indent=2) + '\n')


if __name__ == '__main__':
    from cumem_none_ref_compat import install
    lock = json.loads(Path(__file__).with_name('ja_nonefixed_runtime_20261020.CONFIG.json').read_text())
    install(lock['binary_sha256'])
    import prompt_budget_guard
    prompt_budget_guard.install()
    prefix.seeded.original_init = init
    prefix.main()
