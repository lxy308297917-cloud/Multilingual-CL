"""Resolve random-batch BF16 variation using original teacher batch shapes."""
import os,sys,time,argparse,subprocess
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from settings import *
sys.path.insert(0,str(OLD/'representation_alignment_v1'));from common import pack,pool
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name');ap.parse_args();start=time.time();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    uuid=os.environ['CUDA_VISIBLE_DEVICES'];pids=subprocess.check_output(['nvidia-smi','--id='+uuid,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines();assert all(int(p)==os.getpid() for p in pids if p.strip())
    teacher=torch.load(TRAIN/'EN_TEACHER.pt',map_location='cpu',weights_only=False);rr=rows(OLD/'representation_alignment_v1/data/train.jsonl');selected=sorted(range(len(rr)),key=lambda i:hashlib.sha256(('audit-train20261022:'+rr[i]['id']).encode()).hexdigest())[:8]
    tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(BASE,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();result=[]
    with torch.inference_mode():
        for i in selected:
            startidx=i//8*8;batch=rr[startidx:startidx+8];x=pack([r['en_ids'] for r in batch],tok.pad_token_id);v=pool(model.model(**x,output_hidden_states=True,use_cache=False).hidden_states[10],x['attention_mask']).cpu()[i-startidx];expected=teacher['vectors'][i];result.append({'max_abs':float((v-expected).abs().max()),'relative_L2':float((v-expected).norm()/expected.norm()),'bitwise_equal':torch.equal(v,expected)})
    # Same eight IDs regrouped, distinct from their original cache batch shapes.
    x=pack([rr[i]['en_ids'] for i in selected],tok.pad_token_id)
    with torch.inference_mode():v=pool(model.model(**x,output_hidden_states=True,use_cache=False).hidden_states[10],x['attention_mask']).cpu()
    e=teacher['vectors'][selected];regrouped={'max_abs':float((v-e).abs().max()),'mean_relative_L2':float(((v-e).norm(dim=1)/e.norm(dim=1)).mean()),'mean_cosine_difference':float((1-torch.nn.functional.cosine_similarity(v,e,dim=1)).mean())}
    write(REPORT/'TEACHER_ACTUAL_BATCH_IDENTITY.json',{'original_batch_shape_replay':result,'regrouped_eight':regrouped,'all_original_shape_bitwise_equal':all(x['bitwise_equal'] for x in result),'teacher_sha256':sha(TRAIN/'EN_TEACHER.pt'),'model_sha256':sha(BASE/'model.safetensors'),'wall_seconds':time.time()-start,'gpu_hours':(time.time()-start)/3600,'note':'Original cache identity is tested with identical source batch size8 and right-padding shapes; regrouped BF16 SDPA variation is separately measured, not treated as row mismatch.'})
    print('teacher batch identity',all(x['bitwise_equal'] for x in result),regrouped,flush=True)
if __name__=='__main__':main()
