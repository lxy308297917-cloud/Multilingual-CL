"""Existing models only, no parameter writeback. Fixed192 fresh source groups."""
import argparse,os,time,subprocess,traceback,sys
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from settings import *
from scoring_v2 import score,behavior

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',choices=list(MODELS),required=True);a=ap.parse_args();out=RUN/'generation'/a.name;out.mkdir(parents=True,exist_ok=False);start=time.time();records=[]
    try:
        protocol=json.loads((RUN/'PROTOCOL_FROZEN.json').read_text());accept=json.loads((RUN/'ACCEPTANCE_FROZEN.json').read_text());assert sha(pathlib.Path(__file__).parent/'scoring_v2.py')==accept['scorer_sha256'];data=json.loads((RUN/'DATA_FROZEN.json').read_text());part=data['partitions']['comparison'];assert sha(part['path'])==part['sha256']
        uuid=os.environ['CUDA_VISIBLE_DEVICES'];assert uuid.startswith('GPU-')
        def guard():
            ps=subprocess.check_output(['nvidia-smi','--id='+uuid,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
            assert all(int(p)==os.getpid() for p in ps if p.strip()),'Other GPU compute appeared'
        guard();torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(MODELS[a.name],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
        eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);allrows=rows(part['path']);counts={'prompt_tokens':0,'generated_tokens':0};physical=[]
        write(out/'CONFIG.json',{'name':a.name,'model':MODELS[a.name],'model_sha256':sha(pathlib.Path(MODELS[a.name])/'model.safetensors'),'scorer_sha256':accept['scorer_sha256'],'protocol_sha256':sha(RUN/'PROTOCOL_FROZEN.json'),'data_sha256':part['sha256'],'gpu_uuid':uuid,'started_unix':start,'code_sha256':sha(__file__)})
        with (out/'PREDICTIONS.jsonl').open('w') as f,torch.inference_mode():
            for task in ('sw_reading','en_reading','en_math'):
                rr=[r for r in allrows if r['task']==task];cap=protocol['generation_caps'][task]
                for begin in range(0,len(rr),4):
                    guard();assert time.time()<protocol['hard_finish_unix'],'Frozen end deadline reached'
                    batch=rr[begin:begin+4];prompts=[]
                    for r in batch:
                        msgs=([{'role':'system','content':r['system']}] if r.get('system') else [])+[{'role':'user','content':r['prompt']}]
                        prompts.append(tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True))
                    width=max(map(len,prompts));xx={'input_ids':torch.tensor([[tok.pad_token_id]*(width-len(p))+p for p in prompts],device='cuda'),'attention_mask':torch.tensor([[0]*(width-len(p))+[1]*len(p) for p in prompts],device='cuda')}
                    yy=model.generate(**xx,max_new_tokens=cap,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=sorted(eos),use_cache=True)[:,width:].cpu().tolist()
                    for r,ids,pr in zip(batch,yy,prompts):
                        end=next((i+1 for i,v in enumerate(ids) if v in eos),len(ids));ids=ids[:end];text=tok.decode(ids,skip_special_tokens=True).strip()
                        try:
                            from langdetect import detect,DetectorFactory
                            DetectorFactory.seed=0;lang=detect(text)
                        except Exception:lang='und'
                        rec={'id':r['id'],'group_id':r['group_id'],'task':task,'model':a.name,'text':text,'token_ids':ids,'prompt_tokens':len(pr),'generated_tokens':len(ids),'capped':not ids or ids[-1] not in eos,'ending_token_id':ids[-1] if ids else None,'language_diagnostic':lang,'scores':score(r,text),'behavior':behavior(text)}
                        records.append(rec);f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush();counts['prompt_tokens']+=len(pr);counts['generated_tokens']+=len(ids)
                    physical.append(int(subprocess.check_output(['nvidia-smi','--id='+uuid,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()))
                    write(out/'PROGRESS.json',{'completed':len(records),'task':task,'elapsed_seconds':time.time()-start});del xx,yy
        assert len(records)==192 and len({r['id'] for r in records})==192
        write(out/'DONE.json',{'name':a.name,'rows':len(records),'counts':counts,'wall_seconds':time.time()-start,'gpu_hours':(time.time()-start)/3600,'physical_peak_mib_observed':max(physical),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'predictions_sha256':sha(out/'PREDICTIONS.jsonl'),'not_formal_test':True})
    except BaseException as e:
        write(out/'FAILED.json',{'error':str(e),'traceback':traceback.format_exc(),'completed':len(records),'elapsed_seconds':time.time()-start});raise
if __name__=='__main__':main()
