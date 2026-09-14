"""Read-only check of every saved weight against Base and the exported selection mask."""
import argparse,json,hashlib
from pathlib import Path
import torch,numpy as np
from safetensors import safe_open
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--base',type=Path,default=Path('/root/models/Qwen2.5-1.5B-Instruct'));a=p.parse_args();torch.set_num_threads(1)
def index(root):
 out={}
 for f in root.glob('*.safetensors'):
  with safe_open(f,framework='pt',device='cpu') as sf:
   for key in sf.keys():
    if key in out:raise ValueError('duplicate tensor')
    out[key]=f
 return out
sel=a.model/'selection';rec=json.loads((sel/'MASK_DONE.json').read_text());maskfile=sel/'masks.npz'
assert hashlib.sha256(maskfile.read_bytes()).hexdigest()==rec['mask_sha256']
masks=np.load(maskfile,allow_pickle=False);rows={x['name']:x for x in rec['parameters']};base=index(a.base);model=index(a.model)
assert set(base)==set(model)==set(rows),'Weight/mask names differ'
results=[]
for name in sorted(model):
 r=rows[name]
 with safe_open(base[name],framework='pt',device='cpu') as bf,safe_open(model[name],framework='pt',device='cpu') as mf:
  x=bf.get_tensor(name);y=mf.get_tensor(name)
  assert x.shape==y.shape and x.dtype==y.dtype and list(y.shape)==r['shape']
  assert torch.isfinite(y).all(),name
  changed=(x!=y).reshape(-1);total=int(changed.sum())
  if r['mode']=='all':frozen=0
  elif r['mode']=='none':frozen=total
  else:
   allowed=torch.from_numpy(np.unpackbits(masks[r['key']],count=y.numel()).astype(bool))
   assert int(allowed.sum())==r['eligible_elements']
   frozen=int((changed & ~allowed).sum())
  results.append({'name':name,'changed':total,'frozen_changed':frozen,'eligible':r['eligible_elements']})
frozen=sum(x['frozen_changed'] for x in results);changed=sum(x['changed'] for x in results)
out={'passed':frozen==0 and changed>0,'model':str(a.model),'changed_elements':changed,'frozen_changed_elements':frozen,'parameters':results,'scope':'all saved static-mask weights; MoFO dynamic per-step selection is not verified by this check','training':json.loads((a.model/'TRAINING_DONE.json').read_text())}
(a.model/'weight_update_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='parameters'}));assert out['passed']
