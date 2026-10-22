"""Frozen paired mastery sampling and full-CE/margin diagnostics, no test data."""
import hashlib,json,time
from pathlib import Path
import torch
from core import calibration_context
from mastery import MasteryCache,extract_answer,numeric_value
from prepare_calibration import SYSTEM
ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def model_digest(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());flat=p.detach().view(-1)
        for start in range(0,flat.numel(),1<<20):h.update(flat[start:start+(1<<20)].cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def write(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def prompts(tokenizer,rows):
    return [(row,lang,tokenizer.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':row['question_'+lang]}],tokenize=True,add_generation_prompt=True)) for row in rows for lang in ('en','sw')]
def left_batch(items,pad,device):
    width=max(map(len,items));return {'input_ids':torch.tensor([[pad]*(width-len(ids))+ids for ids in items],device=device),'attention_mask':torch.tensor([[0]*(width-len(ids))+[1]*len(ids) for ids in items],device=device)}

def collect_mastery(model,tokenizer,rows,*,step,output,cache=None,batch_size=4,diagnostics=False,deadline=None,resource_check=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    def check_limits():
        if deadline is not None and time.monotonic()>=deadline:raise RuntimeError('Frozen GPU-time budget reached during mastery/proxy stage')
        if resource_check is not None:resource_check()
    check_limits()
    manifest=ROOT/'data/calibration_manifest.v1.json';identity=sha(manifest);prompt_hash=hashlib.sha256(SYSTEM.encode()).hexdigest()
    cache=cache or MasteryCache(identity,prompt_hash)
    if cache.identity!={'data_sha256':identity,'prompt_sha256':prompt_hash}:raise ValueError('Mastery identity changed')
    if cache.round+1!=step//512:raise ValueError('Nonsequential cache schedule')
    device=next(model.parameters()).device;jobs=prompts(tokenizer,rows);results={row['id']+':'+lang:{'gold':row['gold'],'outputs':[]} for row,lang,ids in jobs}
    eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);pad=tokenizer.pad_token_id
    counts={'generation_prompt_tokens':0,'generated_tokens':0,'proxy_prompt_tokens':0,'proxy_supervised_tokens':0};ledger=[]
    config={'step':step,'rows':len(rows),'languages':['en','sw'],'manifest_sha256':identity,'prompt_sha256':prompt_hash,'batch_size':batch_size,'greedy':1,'sample_seeds':[17,18,19],'temperature':.7,'top_p':1.,'top_k':0,'max_new_tokens':512,'output_language_requested':'English in both input languages','global_rng_isolated':True,'model_sha256':model_digest(model),'code_sha256':sha(__file__),'parser_sha256':sha(ROOT/'mastery.py'),'scope':'training-side mastery and proxy diagnostic, not dev/formal evaluation'}
    write(output/'CONFIG.json',config)
    with calibration_context(model),torch.inference_mode(),(output/'GENERATIONS.jsonl').open('w') as stream:
        for trial,seed in enumerate([None,17,18,19]):
            for begin in range(0,len(jobs),batch_size):
                check_limits()
                batch=jobs[begin:begin+batch_size];tag='|'.join(r['id']+':'+l for r,l,ids in batch)
                actual_seed=int.from_bytes(hashlib.sha256((str(seed)+':'+tag).encode()).digest()[:4],'big')
                if seed is not None:torch.manual_seed(actual_seed)
                x=left_batch([ids for r,l,ids in batch],pad,device)
                kwargs={'max_new_tokens':512,'do_sample':seed is not None,'use_cache':True,'pad_token_id':pad,'eos_token_id':sorted(eos)}
                if seed is not None:kwargs.update(temperature=.7,top_p=1.,top_k=0,repetition_penalty=1.)
                if deadline is not None:kwargs['max_time']=max(.001,deadline-time.monotonic())
                generated=model.generate(**x,**kwargs)[:,x['input_ids'].shape[1]:].cpu().tolist()
                check_limits()
                for (row,lang,ids),tokens in zip(batch,generated):
                    end=next((i+1 for i,t in enumerate(tokens) if t in eos),len(tokens));tokens=tokens[:end];text=tokenizer.decode(tokens,skip_special_tokens=True).strip();capped=not tokens or tokens[-1] not in eos
                    pred,parser=extract_answer(text)
                    try:
                        from langdetect import DetectorFactory,detect
                        DetectorFactory.seed=0;language_tag=detect(text)
                    except Exception:language_tag='und'
                    record={'id':row['id'],'language':lang,'step':step,'kind':'greedy' if seed is None else 'sample','seed':seed,'batch_seed':actual_seed if seed is not None else None,'text':text,'token_ids':tokens,'generated_tokens':len(tokens),'prompt_tokens':len(ids),'capped':capped,'parser':parser,'language_diagnostic':language_tag,'language_diagnostic_method':'langdetect seed0; not a correctness/filtering criterion','correct':pred is not None and pred==numeric_value(row['gold'])}
                    results[row['id']+':'+lang]['outputs'].append(record);stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush();ledger.append(record)
                    counts['generation_prompt_tokens']+=len(ids);counts['generated_tokens']+=len(tokens)
                del x,generated
    known=cache.refresh(results,round_id=step//512,model_identity=config['model_sha256']);write(output/'MASTERY.json',cache.state_dict())
    if diagnostics:
        proxy=[]
        with calibration_context(model),torch.inference_mode():
            def score(prefix,suffix):
                context=tokenizer.decode(prefix,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                ids=tokenizer(context+suffix,add_special_tokens=False)['input_ids']
                if ids[:len(prefix)]!=prefix:raise ValueError('Unstable teacher-forcing token boundary')
                answer=ids[len(prefix):];x=torch.tensor([ids],device=device)
                logits=model(input_ids=x,use_cache=False).logits[0,len(prefix)-1:len(prefix)+len(answer)-1].float()
                logp=logits.log_softmax(-1).gather(1,torch.tensor(answer,device=device)[:,None]).squeeze(1)
                counts['proxy_prompt_tokens']+=len(prefix);counts['proxy_supervised_tokens']+=len(answer)
                return {'sum_logp':logp.sum().item(),'mean_logp':logp.mean().item(),'answer_tokens':len(answer)}
            for row,lang,prefix in jobs:
                check_limits()
                full=score(prefix,row['reference_answer']+'<|im_end|>');entry={'id':row['id'],'language':lang,'full_reference_CE':-full['mean_logp'],'full_reference_tokens':full['answer_tokens'],'mastered':known[row['id']+':'+lang],'views':{}}
                for mode,hint in [('natural_assistant_start',''),('forced_final_prefix','The final answer is:\n')]:
                    context=tokenizer(tokenizer.decode(prefix,skip_special_tokens=False,clean_up_tokenization_spaces=False)+hint,add_special_tokens=False)['input_ids']
                    positive=score(context,'\\boxed{'+row['gold']+'}<|im_end|>');negative=score(context,'\\boxed{'+row['negative_answer']+'}<|im_end|>')
                    entry['views'][mode]={'positive':positive,'negative':negative,'sum_margin':positive['sum_logp']-negative['sum_logp'],'length_normalized_margin':positive['mean_logp']-negative['mean_logp'],'candidate_numeric_equivalence':False}
                proxy.append(entry)
        write(output/'PROXY_DIAGNOSTICS.json',{'scope':'CE vs sum/normalized answer margin and forced-prefix sensitivity; training-side diagnostic only, no automatic proxy selection','rows':proxy})
    summary={'step':step,'known_by_language':{lang:sum(known[r['id']+':'+lang] for r in rows) for lang in ('en','sw')},'joint_known':sum(known[r['id']+':en'] and known[r['id']+':sw'] for r in rows),'counts':counts,'wall_seconds':time.monotonic()-started,'cache_sha256':sha(output/'MASTERY.json'),'config_sha256':sha(output/'CONFIG.json'),'no_formal_or_dev_examples':True}
    write(output/'DONE.json',summary);return cache,summary
