"""Single preregistered unused-development endpoint; never loads formal tests."""
import argparse,json,os,time,traceback
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from calibration import ROOT,sha,write,left_batch
from train import check_compute_owner,BASE
from budget import stage_deadline,verify_software
from development_scoring import score_record,summarize

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--label',choices=['Base','A2048','D2048','R2048','E2048','F2048','S2048'],required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();records=[];counts={'prompt':0,'generated':0};physical=[]
    try:
        protocol=json.loads((ROOT/'comparison_protocol.v1.json').read_text());release=json.loads((ROOT/'DEVELOPMENT_RELEASE.json').read_text())
        verify_software(release)
        if release['status']!='released' or release['protocol_sha256']!=sha(ROOT/'comparison_protocol.v1.json'):raise RuntimeError('Development release mismatch')
        for path,expected in {**protocol['frozen_files'],**release['code_hashes']}.items():
            if sha(path)!=expected:raise RuntimeError('Frozen input/code changed: '+path)
        deadline=stage_deadline(start,protocol['budget']['per_development_model_gpu_hours']*3600)
        model_path=Path(args.model)
        if args.label=='Base':
            if model_path.resolve()!=BASE.resolve():raise RuntimeError('Wrong Base model')
        else:
            receipt=json.loads((model_path.parent/'TRAINING_DONE.json').read_text())
            if receipt['method']+'2048'!=args.label or receipt['global_step']!=2048 or receipt['protocol_sha256']!=sha(ROOT/'comparison_protocol.v1.json'):raise RuntimeError('Only frozen2048 complete endpoints')
            for path,expected in receipt['final_weights'].items():
                if sha(path)!=expected:raise RuntimeError('Trained model identity changed')
        if not os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-') or torch.cuda.device_count()!=1:raise RuntimeError('Single queue-assigned UUID required')
        torch.set_num_threads(4);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;check_compute_owner()
        tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
        config={'label':args.label,'model':str(model_path),'weights':{str(p):sha(p) for p in model_path.glob('*.safetensors')},'protocol_sha256':sha(ROOT/'comparison_protocol.v1.json'),'release_sha256':sha(ROOT/'DEVELOPMENT_RELEASE.json'),'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'batch_size':4,'greedy':True,'no_formal_test':True,'started_unix':time.time()};write(out/'CONFIG.json',config)
        manifest=json.loads(Path(protocol['development']['manifest']).read_text());eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
        with (out/'PREDICTIONS.jsonl').open('w') as stream,torch.inference_mode():
            for partition,info in manifest['partitions'].items():
                rows=[json.loads(s) for s in Path(info['path']).read_text().splitlines()];cap=protocol['development']['max_new_tokens'][partition]
                for begin in range(0,len(rows),4):
                    check_compute_owner()
                    if time.monotonic()>=deadline:raise RuntimeError('Frozen per-model development GPUh ceiling reached')
                    batch=rows[begin:begin+4];prompts=[]
                    for row in batch:
                        messages=([{'role':'system','content':row['system']}] if row.get('system') else [])+[{'role':'user','content':row['prompt']}]
                        prompts.append(tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))
                    if max(map(len,prompts))+cap>model.config.max_position_embeddings:raise RuntimeError('Frozen prompt exceeds model context; no truncation permitted')
                    x=left_batch(prompts,tok.pad_token_id,'cuda');generated=model.generate(**x,max_new_tokens=cap,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=sorted(eos),max_time=max(.001,deadline-time.monotonic()))[:,x['input_ids'].shape[1]:].cpu().tolist()
                    if time.monotonic()>deadline:raise RuntimeError('Frozen dev time budget reached during generation; partial outputs not complete scores')
                    for row,prompt,tokens in zip(batch,prompts,generated):
                        end=next((i+1 for i,t in enumerate(tokens) if t in eos),len(tokens));tokens=tokens[:end];text=tok.decode(tokens,skip_special_tokens=True).strip()
                        try:
                            from langdetect import DetectorFactory,detect
                            DetectorFactory.seed=0;language=detect(text)
                        except Exception:language='und'
                        record={'id':row['id'],'group_id':row['group_id'],'task':row['task'],'partition':partition,'label':args.label,'text':text,'token_ids':tokens,'generated_tokens':len(tokens),'prompt_tokens':len(prompt),'capped':not tokens or tokens[-1] not in eos,'ending_token_id':tokens[-1] if tokens else None,'language_diagnostic':language,'scores':score_record(partition,row,text)}
                        records.append(record);stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush();counts['prompt']+=len(prompt);counts['generated']+=len(tokens)
                    import subprocess
                    physical.append(int(subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()))
                    write(out/'PROGRESS.json',{'partition':partition,'completed':len(records),'total':sum(p['rows'] for p in manifest['partitions'].values()),'elapsed_seconds':time.monotonic()-start,'counts':counts})
                    del generated,x
        write(out/'DONE.json',{'label':args.label,'scores':summarize(records),'counts':counts,'predictions_sha256':sha(out/'PREDICTIONS.jsonl'),'wall_seconds':time.monotonic()-start,'physical_peak_mib_observed':max(physical),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'protocol_sha256':sha(ROOT/'comparison_protocol.v1.json'),'not_formal_benchmark':True})
    except BaseException as e:write(out/'FAILED.json',{'error':str(e),'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-start,'completed_predictions':len(records),'counts':counts});raise
if __name__=='__main__':main()
