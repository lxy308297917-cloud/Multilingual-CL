"""Bounded semantic QA, no downstream feedback or unrestricted pool expansion."""
import time,traceback,shutil
from common import *
from launch import ledger
from transformers import AutoTokenizer,AutoModelForCausalLM

def main():
 target=OUT/'quality_final_v2';target.mkdir(parents=True,exist_ok=False);begun=time.monotonic();ok=False;input_tokens=0;processed=0;accepted={};cfg={'threshold':.9,'both_orders':True,'train_candidates':3500,'held_candidates':2500,'selected_train':2000,'selected_dev':500,'selected_diagnostic':1000,'max_gpu_hours':.5,'no_downstream_feedback':True,'code_sha256':sha(__file__)};write(target/'CONFIG.json',cfg)
 key='semantic_quality_final_v2'
 def reserve(d):
  used=sum(e.get('gpu_hours',e['reserved_gpu_hours']) for e in d['entries'])
  if used+.5>24:raise RuntimeError('Alignment budget exhausted')
  d['entries'].append({'id':key,'reserved_gpu_hours':.5,'status':'running','gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES']})
 ledger(reserve)
 try:
  pilot=json.loads((OUT/'quality_pilot/DONE.json').read_text());assert pilot['gate_passed'];rrpilot=json.loads((OUT/'quality_pilot/RECORDS.json').read_text());known_bad={rrpilot[i]['id'] for i in [10,13,18,19,20,32,40,45,47,53]};postreview=json.loads((ROOT/'data_accepted_v2/review_samples.json').read_text());known_bad.update(postreview[i]['id'] for i in [0,2,3,8,11,16,17,22,38,42]);role=json.loads((ROOT/'AYA_ROLE_AUDIT.json').read_text());known_bad.update(role['excluded_ids']);cache={r['id']:r for r in rows(OUT/'quality_full_v1/RECORDS.jsonl')};seed();check_compute();modelpath=HOME/'models/Qwen3-8B'
  for name,expected in pilot['model_files'].items():assert sha(modelpath/name)==expected
  tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();yes=tok.encode('YES',add_special_tokens=False);no=tok.encode('NO',add_special_tokens=False);assert len(yes)==len(no)==1
  candidates=ROOT/'data_candidates_v2';pools={'train':rows(candidates/'train.jsonl'),'held':rows(candidates/'dev.jsonl')+rows(candidates/'diagnostic.jsonl')}
  with (target/'RECORDS.jsonl').open('w') as f,torch.inference_mode():
   for pool,rr in pools.items():
    chosen=[];required=2000 if pool=='train' else 1500
    for start in range(0,len(rr),4):
     check_compute()
     if time.monotonic()-begun>1800:raise RuntimeError('Fixed semantic-QA GPUh ceiling')
     batch=rr[start:start+4];texts=[];pending=[];verdicts={}
     for r in batch:
      from difflib import SequenceMatcher
      if SequenceMatcher(None,' '.join(r['en'].casefold().split()),' '.join(r['sw'].casefold().split()),autojunk=False).ratio()>=.65:verdicts[r['id']]={'accepted':False,'reason':'cross_language_character_copy_ratio_ge_.65'};continue
      if r['id'] in known_bad:verdicts[r['id']]={'accepted':False,'reason':'assistant_reviewed_mismatch'};continue
      if r['id'] in cache and cache[r['id']]['en']==r['en'] and cache[r['id']]['sw']==r['sw']:verdicts[r['id']]={**cache[r['id']],'reused_previous_semantic_QA':True};continue
      pending.append(r)
      for reverse in (False,True):
       pair=('Swahili: '+r['sw']+'\nEnglish: '+r['en']) if reverse else ('English: '+r['en']+'\nSwahili: '+r['sw']);prompt='Judge whether these English and Swahili sentences express the SAME meaning. Reject unrelated texts, different participants, actions, numbers, named entities, negation or missing major information. Minor grammar mistakes are acceptable if the meaning is preserved. Reply with exactly YES or NO.\n\n'+pair;texts.append(tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False))
     if texts:
      input_tokens+=sum(map(len,texts));w=max(map(len,texts));x={'input_ids':torch.tensor([[tok.pad_token_id]*(w-len(z))+z for z in texts],device='cuda'),'attention_mask':torch.tensor([[0]*(w-len(z))+[1]*len(z) for z in texts],device='cuda')};logits=model.lm_head(model.model(**x,use_cache=False).last_hidden_state[:,-1]).float();prob=logits[:,[no[0],yes[0]]].softmax(-1)[:,1].cpu().tolist();top=logits.argmax(-1).cpu().tolist()
      for j,r in enumerate(pending):verdicts[r['id']]={'yes_probabilities':prob[j*2:j*2+2],'accepted':min(prob[j*2:j*2+2])>=.9,'top_tokens':[tok.decode([z]) for z in top[j*2:j*2+2]]}
      del x,logits
     for r in batch:
      v=verdicts[r['id']];record={'id':r['id'],'pool':pool,'en':r['en'],'sw':r['sw'],**v};f.write(json.dumps(record,ensure_ascii=False)+'\n');processed+=1
      if v['accepted'] and len(chosen)<required:chosen.append({**r,'semantic_qa':v})
     f.flush();write(target/'PROGRESS.json',{'pool':pool,'selected':len(chosen),'required':required,'processed_total':processed,'input_tokens':input_tokens,'elapsed_seconds':time.monotonic()-begun})
     if len(chosen)==required:break
    if len(chosen)!=required:raise RuntimeError('Predeclared candidate ceiling cannot fill quality pool: '+pool+' '+str(len(chosen)))
    accepted[pool]=chosen
  final=ROOT/'data_accepted_v3';final.mkdir(exist_ok=False)
  for name in ('main_manifest.json','main_prefix.json'):shutil.copy2(candidates/name,final/name)
  for name,rr in [('train',accepted['train']),('dev',accepted['held'][:500]),('diagnostic',accepted['held'][500:])]:
   with (final/(name+'.jsonl')).open('w') as f:
    for r in rr:f.write(json.dumps(r,ensure_ascii=False)+'\n')
  write(final/'manifest.json',{'source_candidates_manifest_sha256':sha(candidates/'manifest.json'),'candidates_manifest':str(candidates/'manifest.json'),'quality_records_sha256':sha(target/'RECORDS.jsonl'),'aya_role_audit_sha256':sha(ROOT/'AYA_ROLE_AUDIT.json'),'copy_exclusion':'casefold whitespace-normalized SequenceMatcher ratio>=.65; conservative removal including possible proper-name-dominated valid pairs; identical B/C/D pool','manual_known_mismatch_ids':sorted(known_bad),'quality_code_sha256':sha(__file__),'splits':{k:{'pairs':n,'sha256':sha(final/(k+'.jsonl'))} for k,n in [('train',2000),('dev',500),('diagnostic',1000)]},'quality_limits':'Model-assisted semantic QA and assistant post-review, not independent native gold; filtering may introduce domain/style bias. Fixed same pool in B/C/D. Main stream unchanged including its pre-existing noise.','near_exclusion':'Inherited numeric-masked char3-5gram TFIDF<.80 against formal/development+all main MT and within/across heldout; no proof of semantic decontamination','semantic_qa':cfg})
  rank=lambda r:hashlib.sha256(('postreview42:'+r['id']).encode()).hexdigest();write(final/'review_samples.json',[{k:r[k] for k in ('id','en','sw')} for pool in ['train','held'] for r in sorted(accepted[pool],key=rank)[:30]])
  write(target/'DONE.json',{'elapsed_seconds':time.monotonic()-begun,'input_tokens':input_tokens,'generated_tokens':0,'processed_pairs':processed,'selected_train':2000,'selected_held':1500,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'data_ready_for_postreview':str(final),'not_yet_training_approved':True});ok=True
 except BaseException as e:write(target/'FAILED.json',{'error':str(e),'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-begun});raise
 finally:
  def finish(d):
   entry=next(e for e in d['entries'] if e['id']==key);entry['gpu_hours']=(time.monotonic()-begun)/3600;entry['status']='complete' if ok else 'failed'
  ledger(finish)
if __name__=='__main__':main()
