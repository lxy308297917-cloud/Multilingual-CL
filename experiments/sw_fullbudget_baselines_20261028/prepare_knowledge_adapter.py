"""Add metadata-only IDs to fewshot split; preserve every scored text and label."""
from settings import *
source=RUN/'evaluation_inputs/knowledge';dest=RUN/'evaluation_inputs/knowledge_adapter'
assert not dest.exists();count=0;hashes={}
for p in sorted(source.glob('*/*/*.json')):
 rows=read(p);relative=p.relative_to(source);target=dest/relative
 original=[(r['question'],r['choices'],r['answer'],r['subject']) for r in rows]
 if p.stem=='validation':rows=[dict(r,id=f'{relative.parts[1]}/validation/{i}') for i,r in enumerate(rows)]
 assert all(set(r)=={'id','subject','question','choices','answer'} for r in rows)
 assert [(r['question'],r['choices'],r['answer'],r['subject']) for r in rows]==original
 write(target,rows);hashes[str(relative)]=sha(target);count+=len(rows)
write(REPORT/'KNOWLEDGE_ADAPTER_ACCEPTANCE.json',dict(passed=True,files=len(hashes),total_rows=count,
 scored_test_rows=28084,fewshot_rows=570,source=str(source),adapter=str(dest),
 changed_fields='validation metadata id only; no prompt/question/options/answer/order/count changes',
 reason='datasets5 requires matching columns in local JSON test and validation splits',hashes=hashes,created_unix=time.time()))
print('Knowledge split schemas aligned without changing scored data')
