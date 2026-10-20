"""Seeded official-core continuation with process-local None ownership repair."""
import json
from pathlib import Path
from cumem_none_ref_compat import install


def main():
    directory=Path(__file__).parent
    lock=json.loads((directory/'te_none_ref_resume_runtime_20261020.CONFIG.json').read_text())
    repair=install(lock['binary_sha256'])
    import seeded_cost_batch_entry as seeded
    original=seeded.original_init

    def init(self,*args,**kwargs):
        original(self,*args,**kwargs)
        assert self.jsd_token_clip==.05 and self.fixed_teacher and self.beta==0
        p=Path(self.args.output_dir);p.mkdir(parents=True,exist_ok=True)
        (p/'ALLOCATOR_COMPAT_ACTUAL.json').write_text(json.dumps(repair,indent=2)+'\n')

    seeded.original_init=init
    seeded.main()


if __name__=='__main__':main()
