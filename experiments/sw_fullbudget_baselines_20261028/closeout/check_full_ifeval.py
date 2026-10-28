"""CPU-only checker coverage over the frozen541 prompts; no model generation."""
from pathlib import Path
import sys,json,time,collections
CODE=Path(__file__).resolve().parents[1]/'code';sys.path.insert(0,str(CODE))
from settings import *
from scoring import instruction
p=read(REPORT/'EVALUATION_FROZEN.json')['sources']['ifeval'];assert sha(p['path'])==p['sha256']
rows=[json.loads(s) for s in Path(p['path']).read_text().splitlines()];assert len(rows)==541
failures=[];types=collections.Counter();total=0
for r in rows:
 types.update(r['instruction_id_list']);total+=len(r['instruction_id_list'])
 try:
  v=instruction(r,'This is a neutral technical checker fixture. It is not a model answer.')
  assert len(v['instruction_pass'])==len(r['instruction_id_list'])
 except Exception as e:failures.append(dict(key=r['key'],types=r['instruction_id_list'],error=repr(e)))
write(REPORT/'FULL_IFEVAL_CHECKER_COVERAGE.json',dict(passed=not failures,prompts=len(rows),instructions=total,
 type_counts=dict(types),failures=failures,source_sha256=p['sha256'],no_generation=True,
 scope='Every frozen instruction ID/kwargs passes checker invocation; not a claim to verify every semantic edge case',completed_unix=time.time()))
assert not failures,failures
print(f'All541 prompt schemas/{total} instructions execute offline without checker errors')
