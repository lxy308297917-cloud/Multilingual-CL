"""Compare a newly saved full checkpoint against Base, including every frozen tensor."""
import argparse,json,re
from pathlib import Path
import torch
from safetensors import safe_open
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--method',choices=['full_fft','layers1_5_23_27_fft'],required=True);a=p.parse_args()
torch.set_num_threads(1)
def index(root):
 result={}
 for path in root.glob('*.safetensors'):
  with safe_open(path,framework='pt',device='cpu') as f:
   for name in f.keys():
    assert name not in result;result[name]=path
 return result
base=index(a.base);new=index(a.model);assert set(base)==set(new),'Parameter names changed'
records=[];frozen_changes=0;selected_changes=0
for name in sorted(base):
 with safe_open(base[name],framework='pt',device='cpu') as f, safe_open(new[name],framework='pt',device='cpu') as g:
  x=f.get_tensor(name);y=g.get_tensor(name);assert x.dtype==y.dtype and x.shape==y.shape
  assert torch.isfinite(y).all(),name
  changed=int(torch.count_nonzero(x!=y));match=re.search(r'layers\.(\d+)\.',name)
  selected=a.method=='full_fft' or (match is not None and int(match[1]) in set(range(1,6))|set(range(23,28)))
  if selected:selected_changes+=changed
  else:frozen_changes+=changed
  records.append({'name':name,'elements':x.numel(),'selected':bool(selected),'changed_elements':changed})
r={'model':str(a.model),'method':a.method,'selected_changed_elements':selected_changes,'frozen_changed_elements':frozen_changes,'passed':selected_changes>0 and frozen_changes==0,'scope':'saved checkpoint weight verification; not downstream capability','training_record':json.loads((a.model/'TRAINING_DONE.json').read_text()) if (a.model/'TRAINING_DONE.json').exists() else None,'parameters':records}
(a.model/'weight_update_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='parameters'}));assert r['passed']
