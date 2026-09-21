"""Non-pickle structured state: JSON tree plus safetensors, hash-bound pair."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file,load_file

def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()

def save_state(path,value):
 path=Path(path);tensors={}
 def encode(x):
  if isinstance(x,torch.Tensor):
   key=str(len(tensors));tensors[key]=x.detach().cpu().contiguous().clone()
   return {'kind':'tensor','key':key}
  if isinstance(x,np.ndarray):
   # NumPy RNG uint32 arrays use an exactly representable supported int64 tensor.
   key=str(len(tensors));tensors[key]=torch.from_numpy(x.astype(np.int64) if x.dtype==np.uint32 else x.copy()).contiguous()
   return {'kind':'numpy','key':key,'dtype':str(x.dtype)}
  if isinstance(x,dict):return {'kind':'dict','items':[[encode(k),encode(v)] for k,v in x.items()]}
  if isinstance(x,(list,tuple)):return {'kind':'tuple' if isinstance(x,tuple) else 'list','items':[encode(v) for v in x]}
  if x is None or type(x) in (str,int,float,bool):return {'kind':'scalar','value':x}
  raise TypeError('Unsupported state type: '+str(type(x)))
 tree=encode(value);tensor_path=path.with_suffix('.safetensors');tmp=tensor_path.with_suffix('.tmp')
 save_file(tensors,str(tmp));os.replace(tmp,tensor_path)
 meta={'format':'structured-safetensors-v1','sha256':file_sha(tensor_path),'tree':tree}
 tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(meta,allow_nan=False));os.replace(tmp,path.with_suffix('.json'))

def load_state(path):
 path=Path(path);meta=json.loads(path.with_suffix('.json').read_text());tp=path.with_suffix('.safetensors')
 if meta['format']!='structured-safetensors-v1' or file_sha(tp)!=meta['sha256']:raise ValueError('State identity mismatch')
 tensors=load_file(str(tp),device='cpu')
 def decode(x):
  k=x['kind']
  if k=='tensor':return tensors[x['key']]
  if k=='numpy':return tensors[x['key']].numpy().astype(x['dtype'])
  if k=='dict':return {decode(a):decode(b) for a,b in x['items']}
  if k in ('list','tuple'):
   items=[decode(v) for v in x['items']];return tuple(items) if k=='tuple' else items
  if k=='scalar':return x['value']
  raise ValueError('Unknown state node')
 return decode(meta['tree'])
