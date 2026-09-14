"""Export the effective binary gradient mask by exercising existing pure hooks on a CPU probe."""
import hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
import torch

def collect(model,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 if (out/'MASK_DONE.json').exists():raise RuntimeError('Mask audit already exists')
 rows=[]
 with zipfile.ZipFile(out/'masks.npz','w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
  for index,(name,p) in enumerate(model.named_parameters()):
   hooks=list((p._backward_hooks or {}).values())
   item={'name':name,'shape':list(p.shape),'elements':p.numel(),'requires_grad':p.requires_grad,'hook_count':len(hooks)}
   if not p.requires_grad:item.update(mode='none',eligible_elements=0)
   elif not hooks:item.update(mode='all',eligible_elements=p.numel())
   else:
    probe=torch.ones(p.shape,dtype=p.dtype,device='cpu')
    for hook in hooks:
     value=hook(probe)
     if value is not None:probe=value
    assert bool(((probe==0)|(probe==1)).all()),'Nonbinary update hook: '+name
    keep=(probe!=0).numpy();key=str(index);packed=np.packbits(keep.reshape(-1));buffer=io.BytesIO();np.save(buffer,packed,allow_pickle=False);archive.writestr(key+'.npy',buffer.getvalue())
    item.update(mode='packed',key=key,eligible_elements=int(keep.sum()))
   rows.append(item)
 record={'parameters':rows,'total_unique_elements':sum(x['elements'] for x in rows),'gradient_eligible_elements':sum(x['eligible_elements'] for x in rows),'observer_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'mask_sha256':hashlib.sha256((out/'masks.npz').read_bytes()).hexdigest(),'scope':'effective binary gradient mask at training start; dynamic optimizer selection is separate'}
 (out/'MASK_DONE.json').write_text(json.dumps(record,indent=2)+'\n');return record
