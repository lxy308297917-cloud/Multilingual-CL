"""Local Qwen3-8B semantic QA pilot, with explicitly assistant-reviewed sentinel pairs."""
import argparse,time,subprocess
from common import *
from transformers import AutoTokenizer,AutoModelForCausalLM

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');a=ap.parse_args();target=OUT/'quality_pilot';target.mkdir(parents=True,exist_ok=False);start=time.monotonic();seed();check_compute();modelpath=HOME/'models/Qwen3-8B';tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
 yes=tok.encode('YES',add_special_tokens=False);no=tok.encode('NO',add_special_tokens=False);assert len(yes)==len(no)==1
 rr=json.loads((ROOT/'data/review_samples.json').read_text());results=[];input_tokens=0
 with torch.inference_mode():
  for start_idx in range(0,len(rr),4):
   check_compute();batch=rr[start_idx:start_idx+4];texts=[]
   for r in batch:
    for reverse in (False,True):
     pair=('Swahili: '+r['sw']+'\nEnglish: '+r['en']) if reverse else ('English: '+r['en']+'\nSwahili: '+r['sw'])
     prompt='Judge whether these English and Swahili sentences express the SAME meaning. Reject unrelated texts, different participants, actions, numbers, named entities, negation or missing major information. Minor grammar mistakes are acceptable if the meaning is preserved. Reply with exactly YES or NO.\n\n'+pair
     texts.append(tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False))
   input_tokens+=sum(map(len,texts));w=max(map(len,texts));x={'input_ids':torch.tensor([[tok.pad_token_id]*(w-len(z))+z for z in texts],device='cuda'),'attention_mask':torch.tensor([[0]*(w-len(z))+[1]*len(z) for z in texts],device='cuda')};logits=model.lm_head(model.model(**x,use_cache=False).last_hidden_state[:,-1]).float();p=logits[:,[no[0],yes[0]]].softmax(-1)[:,1].cpu().tolist();top=logits.argmax(-1).cpu().tolist()
   for j,r in enumerate(batch):results.append({'id':r['id'],'sample_index':start_idx+j,'yes_probabilities':p[j*2:j*2+2],'accepted':min(p[j*2:j*2+2])>=.9,'top_tokens':[tok.decode([v]) for v in top[j*2:j*2+2]],'en':r['en'],'sw':r['sw']})
   write(target/'RECORDS.json',results)
 reject={10,13,18,19,20,32,40,45,47,53};accept={0,2,4,5,8,9,11,12,14,16,17,21,23,24,25,26,28,29,31,33,34,36,37,39,41,42,43,48,49,50,51,54,55,57,58,59};tn=sum(not results[i]['accepted'] for i in reject);tp=sum(results[i]['accepted'] for i in accept)
 write(target/'DONE.json',{'threshold':.9,'both_orders_required':True,'rejected_known_bad':tn,'known_bad':len(reject),'accepted_known_faithful':tp,'known_faithful':len(accept),'gate_passed':tn>=8 and tp>=25,'limits':'Assistant-reviewed sentinels, not independent native gold. Qwen semantic filter may have systematic bias; rejection alone is not semantic correctness proof.','elapsed_seconds':time.monotonic()-start,'input_tokens':input_tokens,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'model_path':str(modelpath),'model_files':{p.name:sha(p) for p in modelpath.glob('*.safetensors')}})
if __name__=='__main__':main()
