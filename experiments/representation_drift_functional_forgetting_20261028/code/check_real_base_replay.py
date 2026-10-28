"""Six inference-only forwards on fixed Base inputs; no extra scientific observations."""
from pathlib import Path
import sys,os,time,json,platform
ROOT=Path(__file__).resolve().parents[3];H=Path('/data0/users/lixiaoyang')
REPORT=ROOT/'docs/results/representation_drift_functional_forgetting_20261028';WORK=ROOT/'work/representation_drift_functional_forgetting_20261028'
sys.path.insert(0,str(ROOT/'outputs/sw_fullbudget_baselines_20261028/code'))
from settings import read,write,sha,BASE
import resources
resources.RUN=WORK
protocol=read(REPORT/'PROTOCOL_FROZEN.json');source=protocol['input_manifest'];assert sha(source['path'])==source['sha256']
rows=[json.loads(line) for line in Path(source['path']).open()]
selected=[0,max(range(len(rows)),key=lambda i:len(rows[i]['input_ids'])),1024]
write(WORK/'BASE_NUMERICAL_REPLAY_SELECTION.json',dict(indices=selected,rule='first row, longest input, first math row',
 fixed_before_replay=True,forward_passes=6,technical_only=True,started_unix=time.time()))
with resources.GPUContext(protocol['execution']['dedicated_uuid'],'drift_Base_numerical_replay01'):
 import torch,numpy as np,random,transformers,sklearn,scipy
 from transformers import AutoModelForCausalLM
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(4)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert sha(BASE/'model.safetensors')==read(REPORT/'SOURCE_IDENTITIES.json')['base_weights_sha256']
 model=AutoModelForCausalLM.from_pretrained(BASE,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
 for p in model.parameters():p.requires_grad_(False)
 cache=np.load(WORK/'Base/first/REPRESENTATIONS.npy',mmap_mode='r')
 predictions=[json.loads(line) for line in (WORK/'Base/first/PREDICTIONS.jsonl').open()]
 current=[None];captured={}
 def hook(index):
  def collect(module,args,result):
   h=result[0] if isinstance(result,tuple) else result;r=current[0]
   positions=torch.tensor(r['content_indices'],device=h.device)
   captured[index]=torch.stack((h[0,r['decision_position']].float(),h[0,r['last_content_position']].float(),h[0].index_select(0,positions).float().mean(0))).cpu().numpy()
  return collect
 handles=[layer.register_forward_hook(hook(i)) for i,layer in enumerate(model.model.layers)]
 evidence=[]
 with torch.inference_mode():
  for index in selected:
   row=rows[index];current[0]=row;ids=torch.tensor([row['input_ids']],device='cuda');mask=torch.ones_like(ids)
   for repeat in range(2):
    captured.clear();result=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
    actual=np.stack([captured[j] for j in range(28)])
    assert np.array_equal(actual,cache[index]),(row['id'],repeat,'representation replay differs',float(np.max(np.abs(actual-cache[index]))))
    if row['task']!='gsm8k':
     scores=result.logits[0,-1,protocol['prompt_protocol']['label_token_ids']].float().log_softmax(0).cpu().numpy()
     assert np.array_equal(scores,np.asarray(predictions[index]['label_logprobs'],dtype=np.float32)),(row['id'],repeat,'logprob differs')
    evidence.append(dict(id=row['id'],repeat=repeat,all28_layers_and3_positions_bitwise_equal=True));del result
 for handle in handles:handle.remove()
 versions=dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,
  numpy=np.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__,cuda_build=torch.version.cuda,
  gpu_name=torch.cuda.get_device_name(0),GPU_UUID=protocol['execution']['dedicated_uuid'])
 write(REPORT/'REAL_BASE_REPLAY_ACCEPTANCE.json',dict(passed=True,observations=evidence,versions=versions,
  no_training=True,no_new_scientific_predictions_or_dataset_change=True,six_technical_forwards=True,
  source_sha256=sha(Path(__file__)),finished_unix=time.time()))
 print('Real Base repeat-cache identity passed.',flush=True)
