"""Compose existing prefix budget, audited JA prompt guard and allocator fix.

No new loss/optimizer/teacher/sampling path. The composed GPU run is validated
by its own saved checkpoint and runtime evidence, not by this module's import.
"""
import json
from pathlib import Path

from cumem_none_ref_compat import install


def main():
    code = Path(__file__).parent
    lock = json.loads((code / 'ja_nonefixed_runtime_20261020.CONFIG.json').read_text())
    repair = install(lock['binary_sha256'])
    import prompt_budget_guard
    prompt_budget_guard.install()
    import prefix20_entry as prefix
    original = prefix.seeded.original_init

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        assert self.jsd_token_clip == .05 and self.fixed_teacher and self.beta == 0
        output = Path(self.args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / 'ALLOCATOR_COMPAT_ACTUAL.json').write_text(json.dumps(repair, indent=2) + '\n')

    prefix.seeded.original_init = init
    prefix.main()


if __name__ == '__main__':
    main()
