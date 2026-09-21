import json,hashlib,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from safetensors.torch import save_file
from activation_bases import estimate_bases,basis_identity
ROOT=Path(__file__).parent
SRC=Path('/root/autodl-tmp/sw_update_utility_v1/calibration_candidates.jsonl')
OUT=Path('/root/autodl-tmp/sw_d2_sparse_projection_v1');OUT.mkdir(exist_ok=True)
rows=[json.loads(s) for s in SRC.open() if json.loads(s)['split']=='scoring']
pools={k:[x for x in rows if x['task']==k] for k in ['instruction','reading','math']}
selected=[pools[k][i] for k,n in [('instruction',86),('reading',85),('math',85)] for i in range(n)]
assert len(selected)==256 and len({x['id'] for x in selected})==256
model=AutoModelForCausalLM.from_pretrained('/root/models/Qwen2.5-1.5B-Instruct',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
model.config.use_cache=False
tok=AutoTokenizer.from_pretrained('/root/models/Qwen2.5-1.5B-Instruct',local_files_only=True)
def collate(items):
 x=items[0];return {'input_ids':torch.tensor([x['input_ids']]),'attention_mask':torch.ones(1,len(x['input_ids']),dtype=torch.long)}
started=time.time();bases,counts=estimate_bases(model,selected,collate)
path=OUT/'edge_bases_rank32.safetensors';save_file(bases,str(path))
record={'calibration_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'selected_ids':[x['id'] for x in selected],'counts':counts,'basis_file':str(path),'basis_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'per_module_sha256':basis_identity(bases),'seconds':time.time()-started}
(ROOT/'BASIS_BUILD.json').write_text(json.dumps(record,indent=2));print(json.dumps({'basis_file':str(path),'basis_sha256':record['basis_sha256'],'seconds':record['seconds']}))
