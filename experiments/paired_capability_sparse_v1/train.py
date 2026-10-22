"""Matched A/D/R official-schedule prefixes with complete resumable state."""
import argparse,hashlib,json,math,os,random,shutil,subprocess,time,traceback,threading
from pathlib import Path
import numpy as np
import torch
from core import SparseProjectedAdamW,calibration_context,pair_weights
from mastery import MasteryCache
from calibration import collect_mastery,write,sha,ROOT
from budget import remaining_seconds,stage_deadline,verify_software
H=Path('/data0/users/lixiaoyang');BASE=H/'models/Qwen2.5-1.5B-Instruct';RECIPE=H/'reproductions/sw_recipe_fft_20261010/data/aya80_mt20'

def learning_rate(step,total=42465,warmup=2124,peak=5e-5):
    if not 0<=step<=total or not 0<warmup<total:raise ValueError('Invalid scheduler position')
    return peak*step/warmup if step<warmup else peak*.5*(1+math.cos(math.pi*(step-warmup)/(total-warmup)))

def per_example_ce(model,rows,pad,device):
    width=max(len(r['input_ids']) for r in rows)
    ids=torch.tensor([r['input_ids']+[pad]*(width-len(r['input_ids'])) for r in rows],device=device)
    mask=torch.tensor([[1]*len(r['input_ids'])+[0]*(width-len(r['input_ids'])) for r in rows],device=device)
    logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
    losses=[]
    for i,r in enumerate(rows):
        target=torch.tensor(r['labels'][1:]+[-100]*(width-len(r['labels'])),device=device);valid=target!=-100
        if not bool(valid.any()):raise ValueError('No shifted supervision')
        losses.append(torch.nn.functional.cross_entropy(logits[i,:-1][valid].float(),target[valid]))
    return torch.stack(losses)

def load_prefix(steps):
    manifest=json.loads((RECIPE/'manifest.json').read_text());indices=json.loads((RECIPE/'indices.json').read_text())[:steps*2];wanted=set(indices);found={}
    if sha(manifest['source_file'])!=manifest['source_sha256']:raise RuntimeError('Main source changed')
    with open(manifest['source_file']) as stream:
        for i,line in enumerate(stream):
            if i in wanted:found[i]=json.loads(line)
            if len(found)==len(wanted):break
    if len(found)!=len(wanted):raise ValueError('Missing main rows')
    return [found[i] for i in indices],indices

def rng_state():return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'python':random.getstate(),'numpy':np.random.get_state()}
def restore_rng(r):
    torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda']);random.setstate(r['python']);np.random.set_state(r['numpy'])
def check_compute_owner():
    gpu=os.environ['CUDA_VISIBLE_DEVICES'];text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    others=[int(line.split(',')[1]) for line in text.splitlines() if line.split(',')[0].strip()==gpu and int(line.split(',')[1])!=os.getpid()]
    if others:raise RuntimeError('Another compute process appeared on assigned GPU; safely exit without touching it: '+str(others))

def save_checkpoint(folder,model,opt,step,cache,counts,protocol_hash,method,spent_seconds):
    if shutil.disk_usage(folder).free<20*1024**3:raise RuntimeError('Insufficient free space for atomic full checkpoint')
    path=folder/f'checkpoint-{step:06d}.pt';temp=path.with_suffix('.pt.tmp')
    if path.exists() or temp.exists():raise FileExistsError('No checkpoint overwrite')
    checkpoint_started=time.monotonic()
    payload={'model':{n:v.detach().cpu() for n,v in model.state_dict().items()},'optimizer':opt.state_dict(),'global_step':step,'mastery':cache.state_dict(),'counts':counts,'rng':rng_state(),'protocol_sha256':protocol_hash,'method':method,'scheduler':{'total':42465,'warmup':2124,'next_lr':learning_rate(step)},'spent_seconds':spent_seconds}
    torch.save(payload,temp);temp.rename(path);digest=sha(path)
    write(path.with_suffix('.json'),{'path':str(path),'sha256':digest,'step':step,'protocol_sha256':protocol_hash,'method':method,'complete_model_optimizer_scheduler_rng_mastery':True,'checkpoint_wall_seconds':time.monotonic()-checkpoint_started})
    return str(path)

def validate_checkpoint_transport(path,model,opt,cache,counts):
    """Real full-trainer persistence/readback; no extra optimizer update or data exposure."""
    started=time.monotonic();path=Path(path);receipt=json.loads(path.with_suffix('.json').read_text())
    if sha(path)!=receipt['sha256']:raise RuntimeError('Checkpoint checksum mismatch')
    state=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    if state['global_step']!=opt.steps or state['counts']!=counts or state['mastery']!=cache.state_dict():raise RuntimeError('Checkpoint position/cache mismatch')
    for name,p in model.state_dict().items():
        if not torch.equal(p.detach().cpu(),state['model'][name]):raise RuntimeError('Model state readback mismatch: '+name)
    for name,entry in opt.state.items():
        for key,value in entry.items():
            if not torch.equal(value.cpu(),state['optimizer']['state'][name][key]):raise RuntimeError('Moment state readback mismatch')
        if not torch.equal(opt.masks[name],state['optimizer']['masks'][name]) or not torch.equal(opt.unions[name],state['optimizer']['unions'][name]):raise RuntimeError('Support state readback mismatch')
    if not torch.equal(torch.get_rng_state(),state['rng']['torch']) or any(not torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),state['rng']['cuda'])):raise RuntimeError('Torch RNG changed during save')
    if random.getstate()!=state['rng']['python'] or any(not np.array_equal(a,b) for a,b in zip(np.random.get_state(),state['rng']['numpy'])):raise RuntimeError('Host RNG changed during save')
    model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);cache.load_state_dict(state['mastery']);restore_rng(state['rng'])
    write(path.parent/f'RECOVERY_TRANSPORT_{opt.steps:06d}.json',{'step':opt.steps,'checkpoint_sha256':receipt['sha256'],'full_model_moments_support_counts_mastery_rng_readback_equal':True,'loader_executed':True,'extra_training_update_or_generation':False,'scope':'full trainer transport check; deterministic next-update replay separately tested in technical run','wall_seconds':time.monotonic()-started})
    receipt['transport_validation_wall_seconds']=time.monotonic()-started;write(path.with_suffix('.json'),receipt)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--method',choices=['A','D','R','E','F','S'],required=True);ap.add_argument('--output',required=True);ap.add_argument('--base-cache',required=True);ap.add_argument('--resume-from');args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);begun=time.monotonic();prior_seconds=0.;writer=None;trace=None
    stop_monitor=threading.Event();physical=[];monitor_errors=[]
    def monitor():
        while not stop_monitor.is_set():
            try:physical.append(int(subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=10).strip()))
            except Exception as error:monitor_errors.append(str(error))
            stop_monitor.wait(5)
    sampler=threading.Thread(target=monitor,daemon=True);sampler.start()
    try:
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from torch.utils.tensorboard import SummaryWriter
        protocol_path=ROOT/'comparison_protocol.v1.json';protocol=json.loads(protocol_path.read_text());protocol_hash=sha(protocol_path)
        release=json.loads((ROOT/'TRAINING_RELEASE.json').read_text())
        verify_software(release)
        if args.method not in release['methods']:raise RuntimeError('Method not released in this phase')
        if release['status']!='released' or release['protocol_sha256']!=protocol_hash:raise RuntimeError('No matching training release')
        for path,expected in {**protocol['frozen_files'],**release['code_hashes']}.items():
            if sha(path)!=expected:raise RuntimeError('Frozen identity changed: '+path)
        if not os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-') or torch.cuda.device_count()!=1:raise RuntimeError('One verified GPU UUID required')
        torch.set_num_threads(4);torch.manual_seed(42);random.seed(42);np.random.seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
        if args.method in ('E','F','S'):
            pairing_release=json.loads((ROOT/'PAIRING_RELEASE.json').read_text())
            if pairing_release['protocol_sha256']!=protocol_hash or pairing_release['status']!='released_complete_matched_block':raise RuntimeError('No budget/strata release for E/F/S block')
        remaining_seconds()
        steps=protocol['training']['steps'];main_rows,indices=load_prefix(steps)
        cal=[json.loads(s) for s in (ROOT/'data/calibration_pairs.v1.jsonl').read_text().splitlines()]
        tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(BASE,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
        model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        named={n:p for n,p in model.named_parameters() if '.layers.' in n and p.ndim==2}
        if len(named)!=196 or sum(p.numel() for p in named.values())!=1310195712:raise RuntimeError('Candidate identity mismatch')
        for n,p in model.named_parameters():p.requires_grad_(n in named)
        opt=SparseProjectedAdamW(named,bf16_policy='reject');device=next(model.parameters()).device
        bc=Path(args.base_cache);base_done=json.loads((bc/'DONE.json').read_text());base_config=json.loads((bc/'CONFIG.json').read_text())
        if base_done['cache_sha256']!=sha(bc/'MASTERY.json') or base_config['manifest_sha256']!=sha(ROOT/'data/calibration_manifest.v1.json'):raise RuntimeError('Base mastery identity changed')
        cache_state=json.loads((bc/'MASTERY.json').read_text());cache=MasteryCache(cache_state['identity']['data_sha256'],cache_state['identity']['prompt_sha256']);cache.load_state_dict(cache_state)
        if sum(cache.records[r['id']+':en']['base'] for r in cal)<8:raise RuntimeError('Insufficient Base mastery')
        start=0;counts={k:0 for k in ('input','supervised','calibration_input','calibration_processed_supervised','calibration_used_supervised','replay_supervised','generated','generation_prompt','proxy_prompt','proxy_supervised')}
        if args.resume_from:
            cp=Path(args.resume_from);receipt=json.loads(cp.with_suffix('.json').read_text())
            borrowed_prefix=args.method in ('E','F','S') and receipt['method']=='D' and receipt['step']==1024
            if sha(cp)!=receipt['sha256'] or receipt['protocol_sha256']!=protocol_hash or (receipt['method']!=args.method and not borrowed_prefix):raise RuntimeError('Resume checkpoint identity changed')
            state=torch.load(cp,map_location='cpu',mmap=True,weights_only=False)
            if state['protocol_sha256']!=protocol_hash or (state['method']!=args.method and not borrowed_prefix):raise RuntimeError('Resume scientific identity')
            if borrowed_prefix:
                if any(r['base'] for key,r in cache.records.items() if key.endswith(':sw')):raise RuntimeError('D/E/F prefix not identical: BaseSW mastery is nonzero')
                joint=sum(state['mastery']['records'][r['id']+':en']['known'] and state['mastery']['records'][r['id']+':sw']['known'] for r in cal)
                en_only=sum(state['mastery']['records'][r['id']+':en']['known'] and not state['mastery']['records'][r['id']+':sw']['known'] for r in cal)
                if joint<8 or en_only<8:raise RuntimeError('Insufficient shared/English-only strata at preregistered1024 fork')
            model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);cache.load_state_dict(state['mastery']);counts=state['counts'];start=state['global_step'];prior_seconds=state['spent_seconds']+receipt.get('checkpoint_wall_seconds',0)+receipt.get('transport_validation_wall_seconds',0)
            if state['scheduler']['next_lr']!=learning_rate(start):raise RuntimeError('Scheduler mismatch')
            if opt.steps!=start or cache.round!=start//512 or start not in (1024,2048):raise RuntimeError('Invalid resumed optimizer/cache/step position')
            if counts['input']!=sum(len(r['input_ids']) for r in main_rows[:2*start]) or counts['supervised']!=sum(sum(v!=-100 for v in r['labels'][1:]) for r in main_rows[:2*start]):raise RuntimeError('Resume token/position mismatch')
            restore_rng(state['rng']);del state
        check_compute_owner();writer=SummaryWriter(str(out/'tensorboard'));trace=(out/'STEPS.jsonl').open('w')
        write(out/'RUN_INFO.json',{'method':args.method,'protocol_sha256':protocol_hash,'release_sha256':sha(ROOT/'TRAINING_RELEASE.json'),'main_indices_sha256':hashlib.sha256(json.dumps(indices,separators=(',',':')).encode()).hexdigest(),'base_cache':str(bc),'base_cache_sha256':sha(bc/'MASTERY.json'),'output':str(out),'resume_from':args.resume_from,'python':os.sys.executable,'torch':torch.__version__,'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'implementation_commit':release['code_commit'],'unused_calibration_compute_disclosed':True,'gradient_transfer':'BF16 raw gradients ->CPU ->FP32; exact value-preserving conversion','calibration_batch_size_per_language':2,'started_at_unix':time.time()})
        cal_encoded={}
        for r in cal:
            for lang in ('en','sw'):
                prefix=tok.apply_chat_template([{'role':'system','content':base_config['system'] if 'system' in base_config else json.loads((ROOT/'data/calibration_manifest.v1.json').read_text())['system']},{'role':'user','content':r['question_'+lang]}],tokenize=True,add_generation_prompt=True)
                answer=tok(r['reference_answer']+'<|im_end|>',add_special_tokens=False)['input_ids'];cal_encoded[r['id']+':'+lang]={'input_ids':prefix+answer,'labels':[-100]*len(prefix)+answer}
        nonzero_lr=32 if start>=33 else max(0,start-1);rejected_first32=0;durations=[];last_checkpoint=None
        for step in range(start,steps):
            deadline=stage_deadline(begun,protocol['budget']['per_training_gpu_hours']*3600-prior_seconds)
            if deadline-time.monotonic()<max(30.,2*max(durations[-20:],default=10.)):raise RuntimeError('Insufficient frozen budget for another full update; preserve partial logs')
            tick=time.monotonic();model.train();model.zero_grad(set_to_none=True);main_loss=0.
            for r in main_rows[step*2:step*2+2]:
                value=per_example_ce(model,[r],tok.pad_token_id,device)[0];(value/2).backward();main_loss+=value.item()/2
                counts['input']+=len(r['input_ids']);counts['supervised']+=sum(v!=-100 for v in r['labels'][1:])
            chosen_indices=[(2*step+j)%len(cal) for j in range(2)];chosen=[cal[i] for i in chosen_indices];guards=[];cal_losses={};cal_weights={};guard_active=False
            known_en=[cache.records[r['id']+':en']['known'] for r in cal];known_sw=[cache.records[r['id']+':sw']['known'] for r in cal]
            pair_mode='paired' if args.method in ('F','S') else 'independent'
            permutation=protocol['pairing']['shuffled_partner_permutation'] if args.method=='S' else None
            w_en,w_sw=pair_weights(known_en,known_sw,pair_mode,permutation)
            with calibration_context(model):
                for lang in ('en','sw'):
                    rr=[cal_encoded[r['id']+':'+lang] for r in chosen];values=per_example_ce(model,rr,tok.pad_token_id,device)
                    use_guard=args.method in ('D','E','F','S') and (lang=='en' or args.method in ('E','F','S'))
                    own=w_en if lang=='en' else w_sw
                    weights=torch.tensor([float(own[i]) if use_guard else 1. for i in chosen_indices],device=device)
                    cal_weights[lang]=weights.detach().cpu().tolist()
                    objective=(values*weights).sum()/weights.sum().clamp_min(1.);g=torch.autograd.grad(objective,list(named.values()));cal_losses[lang]=values.detach().cpu().tolist()
                    if args.method=='R':
                        with torch.no_grad():
                            for p,v in zip(named.values(),g):p.grad.add_(v,alpha=protocol['training']['replay_alpha']/2)
                    cpu={n:v.detach().cpu().float() for (n,p),v in zip(named.items(),g)}
                    if use_guard and bool(weights.sum()>0):guards.append(cpu);guard_active=True
                    del g,cpu,objective,values
                    for i,row in enumerate(rr):
                        used=sum(v!=-100 for v in row['labels'][1:]);counts['calibration_input']+=len(row['input_ids']);counts['calibration_processed_supervised']+=used
                        if use_guard and float(weights[i])>0:counts['calibration_used_supervised']+=used
                        if args.method=='R':counts['replay_supervised']+=used
            norm=torch.nn.utils.clip_grad_norm_(list(named.values()),1.);lr=learning_rate(step);stats=opt.step(lr,guards);del guards
            torch.cuda.synchronize();elapsed=time.monotonic()-tick;durations.append(elapsed)
            stats.update(main_loss=main_loss,calibration_losses=cal_losses,calibration_weights=cal_weights,calibration_ids=[r['id'] for r in chosen],guard_active=guard_active,grad_norm_before_clip=float(norm),update_seconds=elapsed)
            trace.write(json.dumps(stats)+'\n');trace.flush()
            for key,value in stats.items():
                if isinstance(value,(int,float)):writer.add_scalar(key,value,step+1)
            if lr>0 and nonzero_lr<32:
                nonzero_lr+=1;rejected_first32+=int(stats['rejected_rounding'])
                if nonzero_lr==32:
                    projected_seconds=sum(durations)/len(durations)*steps+4*base_done['wall_seconds']+120
                    write(out/'EARLY_COST_GATE.json',{'measured_updates':len(durations),'mean_update_seconds':sum(durations)/len(durations),'projected_full_seconds_including_four_Base_sized_refreshes_and120s_checkpoint_allowance':projected_seconds,'per_training_ceiling_seconds':protocol['budget']['per_training_gpu_hours']*3600,'first32_rejected':rejected_first32,'threshold_rejected':16,'no_dev_scores_used':True})
                    if rejected_first32>16:raise RuntimeError('Predeclared stop: more than50percent BF16 update rejection in first32 nonzero-LR steps')
                    if projected_seconds>protocol['budget']['per_training_gpu_hours']*3600:raise RuntimeError('Measured full-loop cost projection does not fit frozen per-training allocation')
            if step%20==0:check_compute_owner();write(out/'PROGRESS.json',{'step':step+1,'total_steps':steps,'counts':counts,'spent_gpu_hours':(prior_seconds+time.monotonic()-begun)/3600,'recent_seconds_per_update':sum(durations[-20:])/len(durations[-20:]),'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'physical_peak_mib':max(physical) if physical else None})
            model.zero_grad(set_to_none=True)
            if (step+1)%512==0:
                cache,summary=collect_mastery(model,tok,cal,step=step+1,output=out/f'cache_{step+1:06d}',cache=cache,batch_size=4,deadline=stage_deadline(begun,protocol['budget']['per_training_gpu_hours']*3600-prior_seconds),resource_check=check_compute_owner)
                counts['generated']+=summary['counts']['generated_tokens'];counts['generation_prompt']+=summary['counts']['generation_prompt_tokens']
            if (step+1)%1024==0:
                if stage_deadline(begun,protocol['budget']['per_training_gpu_hours']*3600-prior_seconds)-time.monotonic()<120:raise RuntimeError('Insufficient frozen budget for full checkpoint/readback')
                last_checkpoint=save_checkpoint(out,model,opt,step+1,cache,counts,protocol_hash,args.method,prior_seconds+time.monotonic()-begun)
                validate_checkpoint_transport(last_checkpoint,model,opt,cache,counts)
        if stage_deadline(begun,protocol['budget']['per_training_gpu_hours']*3600-prior_seconds)-time.monotonic()<120:raise RuntimeError('Insufficient frozen budget for final model verification')
        final=out/'final';model.save_pretrained(final,safe_serialization=True);tok.save_pretrained(final)
        from safetensors import safe_open
        differences=0
        with safe_open(str(BASE/'model.safetensors'),framework='pt',device='cpu') as original:
            for name,p in model.named_parameters():
                old=original.get_tensor(name);new=p.detach().cpu()
                if name not in named and not torch.equal(old,new):raise RuntimeError('Frozen noncandidate changed: '+name)
                if name in named:differences+=int(torch.count_nonzero(old!=new))
        trace.close();writer.close();write(out/'TRAINING_DONE.json',{'method':args.method,'global_step':opt.steps,'counts':counts,'last_resumable_checkpoint':last_checkpoint,'protocol_sha256':protocol_hash,'final_weights':{str(p):sha(p) for p in final.glob('*.safetensors')},'physical_device_peak_mib':max(physical) if physical else None,'physical_monitor_errors':monitor_errors[-5:],'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'charged_this_execution_gpu_hours':(time.monotonic()-begun)/3600,'total_gpu_hours':(prior_seconds+time.monotonic()-begun)/3600,'final_candidate_values_different_from_base':differences,'noncandidates_exactly_frozen':True,'mean_update_seconds':sum(durations)/len(durations) if durations else None,'base_calibration_shared_cost_not_duplicated':base_done['counts']})
    except BaseException as e:
        write(out/'FAILED.json',{'error':str(e),'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-begun,'prior_seconds':prior_seconds});raise
    finally:
        if trace is not None:trace.close()
        if writer is not None:writer.close()
        stop_monitor.set();sampler.join(timeout=12)
if __name__=='__main__':main()
