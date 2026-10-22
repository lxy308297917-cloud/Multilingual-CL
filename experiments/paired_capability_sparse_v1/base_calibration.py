"""Base mastery/proxy stage; frozen data and limits, no downstream scores."""
import argparse,json,os,time,traceback
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from calibration import collect_mastery,write,sha,ROOT
from core import pair_weights
from train import check_compute_owner
H=Path('/data0/users/lixiaoyang');BASE=H/'models/Qwen2.5-1.5B-Instruct'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 try:
  protocol=json.loads((ROOT/'comparison_protocol.v1.json').read_text())
  if not os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-') or torch.cuda.device_count()!=1:raise RuntimeError('One queue-assigned GPU UUID required')
  audit_path=ROOT/'data/dev_v4/INDEPENDENT_AUDIT.json';audit=json.loads(audit_path.read_text())
  if audit['status']!='passed' or not all(audit['gates'].values()):raise RuntimeError('Dev isolation not passed')
  for path,expected in protocol['frozen_files'].items():
   if sha(path)!=expected:raise RuntimeError('Protocol input changed: '+path)
  torch.set_num_threads(4);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
  tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
  model=AutoModelForCausalLM.from_pretrained(BASE,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
  rows=[json.loads(s) for s in (ROOT/'data/calibration_pairs.v1.jsonl').read_text().splitlines()]
  cache,summary=collect_mastery(model,tok,rows,step=0,output=out/'cache_000000',batch_size=4,diagnostics=True,deadline=start+protocol['budget']['base_mastery_gpu_hours']*3600,resource_check=check_compute_owner)
  en=[cache.records[r['id']+':en']['known'] for r in rows];sw=[cache.records[r['id']+':sw']['known'] for r in rows]
  e=pair_weights(en,sw);f=pair_weights(en,sw,'paired');shuffled=pair_weights(en,sw,'paired',protocol['pairing']['shuffled_partner_permutation'])
  normalized=lambda x:x/x.sum() if x.sum() else x
  write(out/'PAIRING_GATE.json',{'scope':'New1greedy+3samples Base labels; does not preclude later acquiredSW mastery','en_known':sum(en),'sw_known':sum(sw),'joint_known':sum(a and b for a,b in zip(en,sw)),'E_F_weights_equal':all(torch.equal(normalized(a),normalized(b)) for a,b in zip(e,f)),'F_shuffled_weights_equal':all(torch.equal(normalized(a),normalized(b)) for a,b in zip(f,shuffled)),'later_gate':'Assess preregistered D1024 cache; require8joint+8ENonly before E/F/shuffle comparison; prove prefix identity before reuse'})
  if summary['known_by_language']['en']<8:raise RuntimeError('Fewer than8 conservatively mastered English calibration questions; stop before comparative training')
  if time.monotonic()-start>protocol['budget']['base_mastery_gpu_hours']*3600:raise RuntimeError('Base calibration exceeded frozen GPUh ceiling')
  write(out/'DONE.json',{'scope':'Base training-side mastery/CE-margin diagnostic only; no dev/formal efficacy scores','summary':summary,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'wall_seconds':time.monotonic()-start,'protocol_sha256':sha(ROOT/'comparison_protocol.v1.json'),'script_sha256':sha(__file__),'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES']})
 except BaseException as e:write(out/'FAILED.json',{'error':str(e),'traceback':traceback.format_exc(),'wall_seconds':time.monotonic()-start});raise
if __name__=='__main__':main()
