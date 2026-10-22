"""Read-only true-Qwen probes; no training and no checkpoint overwrite."""
import argparse,os,time,sys,subprocess,math,gc
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from settings import *
sys.path.insert(0,str(OLD/'representation_alignment_v1'))
from common import pack,pool,ce,auxiliary,retrieval

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',choices=['Base']+list('ABCD')+[m+'-mid' for m in 'BCD'],required=True);ap.add_argument('--gradients-only',action='store_true');a=ap.parse_args();start=time.time();out=RUN/'mechanism'/a.name;out.mkdir(parents=True,exist_ok=False)
    uuid=os.environ['CUDA_VISIBLE_DEVICES'];torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.use_deterministic_algorithms(True)
    def guard():
        pids=subprocess.check_output(['nvidia-smi','--id='+uuid,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines();assert all(int(p)==os.getpid() for p in pids if p.strip())
        assert time.time()<json.loads((RUN/'PROTOCOL_FROZEN.json').read_text())['hard_finish_unix']
    guard();mid=a.name.endswith('-mid');name=a.name[0] if mid else a.name
    modelpath=TRAIN/'comparison_v3'/name/'checkpoint_002146/model' if mid else pathlib.Path(MODELS[name]);modelhash=sha(modelpath/'model.safetensors')
    tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda();model.config.use_cache=False
    auxrows=rows(OLD/'representation_alignment_v1/data/train.jsonl');teacher=torch.load(TRAIN/'EN_TEACHER.pt',map_location='cpu',weights_only=False)
    assert teacher['ids']==[r['id'] for r in auxrows];assert teacher['layer']==10;assert teacher['train_sha256']==sha(OLD/'representation_alignment_v1/data/train.jsonl');assert teacher['model_sha256']==sha(BASE/'model.safetensors');assert teacher['tokenizer_sha256']==sha(BASE/'tokenizer.json')
    for r in auxrows:
        for lang in ('en','sw'):assert r[lang+'_ids']==tok.encode(r[lang],add_special_tokens=False)
    bank=teacher['vectors'].cuda();rawbank=bank.float();normbank=F.normalize(rawbank,dim=-1);layer=teacher['layer'];identity={'teacher_ids_and_tokenization_all2000_verified':True,'cache_model_base_sha256_verified':True,'hidden_state_index':10,'semantic_layer':'output after decoder block index9; embeddings state0; final normalized state28','padding':'right padded; attention mask excludes padding; inputs are content only with no chat template','teacher_training_sha256':teacher['train_sha256'],'teacher_sha256':sha(TRAIN/'EN_TEACHER.pt'),'model_sha256':modelhash,'dropout':getattr(model.config,'attention_dropout',None),'temperature':.07,'training_bank_rows':len(bank)}
    selected=sorted(range(len(auxrows)),key=lambda i:hashlib.sha256(('audit-train20261022:'+auxrows[i]['id']).encode()).hexdigest())[:64]
    write(out/'IDENTITY.json',identity)
    if not a.gradients_only and not mid:
        model.eval();vectors=[]
        with torch.inference_mode():
            for begin in range(0,64,4):
                guard();ii=selected[begin:begin+4];xx=pack([auxrows[i]['sw_ids'] for i in ii],tok.pad_token_id);hs=model.model(**xx,output_hidden_states=True,use_cache=False).hidden_states;assert len(hs)==29;vectors.append(pool(hs[10],xx['attention_mask']).cpu());del hs,xx
            if name=='Base':
                xx=pack([auxrows[i]['en_ids'] for i in selected[:8]],tok.pad_token_id);actual=pool(model.model(**xx,output_hidden_states=True,use_cache=False).hidden_states[10],xx['attention_mask']).cpu();expected=teacher['vectors'][selected[:8]];identity['teacher_actual_first8_max_abs']=float((actual-expected).abs().max());assert torch.allclose(actual,expected,atol=.05,rtol=.01);del xx
        sw=torch.cat(vectors).float();en=teacher['vectors'][selected].float();nb=normbank.cpu();posidx=torch.tensor(selected);metric={}
        def calc(sw,en,idx,held=False):
            ns=F.normalize(sw.float(),dim=-1);ne=F.normalize(en.float(),dim=-1);pos=(ns*ne).sum(-1)
            logits=ns@nb.T/.07
            if held:
                logits=torch.cat([pos[:,None]/.07,logits],dim=1);labels=torch.zeros(len(sw),dtype=torch.long)
            else:labels=idx
            ret=retrieval(en,sw)
            return {'pairs':len(sw),'fixed_EN_cosine_loss':float((1-pos).mean()),'fixed_bank_InfoNCE':float(F.cross_entropy(logits,labels)),'InfoNCE_candidates':logits.shape[1],'own_positive_in_training_bank':not held,'train_bank_top1':float((logits.argmax(1)==labels).float().mean()),'raw_SW_mean_variance':float(sw.float().var(0).mean()),'raw_EN_mean_variance':float(en.float().var(0).mean()),**ret}
        metric['train64']=calc(sw,en,posidx)
        cache_name='Base' if name=='Base' else 'align-'+name;cached=TRAIN/'diagnostics'/cache_name;cachedmeta=json.loads((cached/'DONE.json').read_text());assert cachedmeta['model_sha256']==modelhash
        held=torch.load(cached/'diagnostic_representations.pt',map_location='cpu',mmap=True,weights_only=True);baseheld=torch.load(TRAIN/'diagnostics/Base/diagnostic_representations.pt',map_location='cpu',mmap=True,weights_only=True)
        hsw=held['sw']['mean'][10].float();hen=baseheld['en']['mean'][10].float();assert hsw.shape==hen.shape==(1000,1536)
        metric['held1000']=calc(hsw,hen,None,True)
        diagrows=rows(OLD/'representation_alignment_v1/data/diagnostic.jsonl')
        with torch.inference_mode():
            xs=pack([r['sw_ids'] for r in diagrows[:8]],tok.pad_token_id);actual=pool(model.model(**xs,output_hidden_states=True,use_cache=False).hidden_states[10],xs['attention_mask']).cpu();expected=hsw[:8];identity['actual_cached_SW_first8_max_abs']=float((actual-expected).abs().max());identity['actual_cached_SW_first8_relative_L2']=float(((actual-expected).norm(dim=1)/expected.norm(dim=1)).mean());assert torch.allclose(actual,expected,atol=.08,rtol=.02);del xs
        torch.save({'selected_train_indices':selected,'ids':[auxrows[i]['id'] for i in selected],'sw_vectors':sw},out/'TRAIN_VECTORS.pt');write(out/'REPRESENTATION.json',metric);write(out/'IDENTITY.json',identity)
        del held,baseheld,hsw,hen,vectors
    # Read-only real gradients at Base and saved middle/end B/C/D; A representation only.
    if name=='A':
        write(out/'DONE.json',{'wall_seconds':time.time()-start,'model':a.name,'read_only':True});return
    model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    # These fixed samples are a hypothetical counterfactual, not the historical next batch.
    manifest=json.loads((OLD/'representation_alignment_v1/data/main_manifest.json').read_text());indices=json.loads((OLD/'representation_alignment_v1/data/main_prefix.json').read_text());wanted=set(indices[6:8]);mainrows={}
    with open(manifest['source']) as f:
        for i,line in enumerate(f):
            if i in wanted:mainrows[i]=json.loads(line)
            if len(mainrows)==len(wanted):break
    mainbatch=[mainrows[i] for i in indices[6:8]]
    order=sorted(range(len(auxrows)),key=lambda i:hashlib.sha256(('aux42:'+auxrows[i]['id']).encode()).hexdigest());ii=order[:8]
    def main_backward():
        total=0.
        for row in mainbatch:
            x=pack([row],tok.pad_token_id,labels=True);labels=x.pop('labels');loss=ce(model(**x,use_cache=False).logits,labels);(loss/2).backward();total+=float(loss.detach())/2
        return total
    def aux_backward(kind,scale):
        total=0.
        if kind=='B':
            rr=[r for i in ii for r in auxrows[i]['ce_rows']]
            for begin in range(0,len(rr),4):
                chunk=rr[begin:begin+4];x=pack(chunk,tok.pad_token_id,labels=True);labels=x.pop('labels');loss=ce(model(**x,use_cache=False).logits,labels);weight=len(chunk)/len(rr);(loss*scale*weight).backward();total+=float(loss.detach())*weight
        else:
            for begin in range(0,8,4):
                jj=ii[begin:begin+4];x=pack([auxrows[i]['sw_ids'] for i in jj],tok.pad_token_id);v=pool(model.model(**x,output_hidden_states=True,use_cache=False).hidden_states[10],x['attention_mask']);loss=auxiliary(v,bank,torch.tensor(jj,device='cuda'),kind);(loss*scale*.5).backward();total+=float(loss.detach())*.5
        return total
    params=list(model.parameters())
    def collect():return [p.grad.detach().cpu().clone() if p.grad is not None else None for p in params]
    model.zero_grad(set_to_none=True);mainloss=main_backward();gm=collect();main_norm=float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach(),2) for p in params if p.grad is not None]),2));model.zero_grad(set_to_none=True)
    state=None;lr=0.;saved_step=0
    if name!='Base':
        cp=modelpath.parent;receipt=json.loads((cp/'RECOVERABLE.json').read_text());assert sha(cp/'state.pt')==receipt['files']['state.pt'];state=torch.load(cp/'state.pt',map_location='cpu',mmap=True,weights_only=False);saved_step=state['step'];lr=state['optimizer']['param_groups'][0]['lr']
    results={}
    for kind in 'BCD':
        guard();model.zero_grad(set_to_none=True);auxloss=aux_backward(kind,1.);ga=collect();model.zero_grad(set_to_none=True);main_backward();aux_backward(kind,.1);gcmb=collect()
        # Exact historical clipping operator applied to this hypothetical combined gradient.
        actual_norm=float(torch.nn.utils.clip_grad_norm_(params,1.));clipped=collect();model.zero_grad(set_to_none=True)
        aa=mm=dot=combo=rounddiff=0.;nonzero=mainzero=0;aux_survive=0;totalcoords=0;block={}
        for n,p,m,g,c in zip([n for n,_ in model.named_parameters()],params,gm,ga,gcmb):
            mf=m.float() if m is not None else torch.zeros_like(p,device='cpu',dtype=torch.float32);gf=g.float() if g is not None else torch.zeros_like(mf);cf=c.float() if c is not None else torch.zeros_like(mf)
            m2=float(mf.square().sum());g2=float(gf.square().sum());mg=float((mf*gf).sum());mm+=m2;aa+=g2;dot+=mg;combo+=float(cf.square().sum());totalcoords+=p.numel();aux_survive+=int(torch.count_nonzero(cf!=mf));rounddiff+=float((cf-(mf+.1*gf)).square().sum())
            import re
            match=re.search(r'layers\.(\d+)\.',n);bucket='layer'+match[1] if match else 'embedding_or_head_or_finalnorm';v=block.setdefault(bucket,{'main_sq':0.,'aux_sq':0.,'dot':0.});v['main_sq']+=m2;v['aux_sq']+=g2;v['dot']+=mg
        result={'main_ce':mainloss,'raw_aux_loss':auxloss,'main_grad_L2_fp32_reduction':math.sqrt(mm),'aux_grad_L2_fp32_reduction':math.sqrt(aa),'scaled_aux_grad_L2':.1*math.sqrt(aa),'scaled_aux_to_main_norm_ratio':.1*math.sqrt(aa/mm),'main_aux_cosine':dot/math.sqrt(mm*aa) if aa>0 else None,'actual_combined_clip_input_norm_BF16':actual_norm,'main_only_clip_input_norm_BF16':main_norm,'combined_clipping_factor':min(1.,1/(actual_norm+1e-6)),'aux_changed_preclip_gradient_coordinates':aux_survive,'total_parameter_coordinates':totalcoords,'BF16_accumulation_difference_L2_vs_main_plus_scaled_raw_aux':math.sqrt(rounddiff),'layer_gradient_sums':block}
        # Replay candidate updates using independent CUDA clones, never model params.
        if state is not None:
            summary={'moment_changed_coordinates':0,'weight_changed_coordinates':0,'main_only_nonzero_weight_displacement':0,'combined_nonzero_weight_displacement':0,'main_only_delta_sq':0.,'combined_delta_sq':0.,'difference_delta_sq':0.,'candidate_coordinates':0}
            sg=state['optimizer']['param_groups'][0];ids=sg['params'];assert len(ids)==len(params)
            with torch.no_grad():
                for p,m,c,paramid in zip(params,gm,clipped,ids):
                    st=state['optimizer']['state'][paramid];outs=[];mom=[]
                    for grad,iscombined in ((m,False),(c,True)):
                        q=torch.nn.Parameter(p.detach().clone());o=torch.optim.AdamW([q],lr=lr,betas=tuple(sg['betas']),eps=sg['eps'],weight_decay=sg['weight_decay'],foreach=True)
                        o.state[q]={'step':st['step'].clone(),'exp_avg':st['exp_avg'].cuda().clone(),'exp_avg_sq':st['exp_avg_sq'].cuda().clone()}
                        q.grad=(grad.cuda().clone() if grad is not None else torch.zeros_like(q))
                        if not iscombined:q.grad.mul_(torch.tensor(min(1.,1/(main_norm+1e-6)),dtype=q.grad.dtype,device='cuda'))
                        o.step();outs.append(q.detach().clone());mom.append(o.state[q]['exp_avg'].clone());del o,q
                    d0=outs[0].float()-p.float();d1=outs[1].float()-p.float();summary['moment_changed_coordinates']+=int(torch.count_nonzero(mom[0]!=mom[1]));summary['weight_changed_coordinates']+=int(torch.count_nonzero(outs[0]!=outs[1]));summary['main_only_nonzero_weight_displacement']+=int(torch.count_nonzero(d0));summary['combined_nonzero_weight_displacement']+=int(torch.count_nonzero(d1));summary['main_only_delta_sq']+=float(d0.square().sum());summary['combined_delta_sq']+=float(d1.square().sum());summary['difference_delta_sq']+=float((d1-d0).square().sum());summary['candidate_coordinates']+=p.numel();del outs,mom,d0,d1
            result['copied_state_AdamW_candidate']=summary
        results[kind]=result;write(out/'GRADIENTS_PARTIAL.json',results);del ga,gcmb,clipped;gc.collect()
    write(out/'GRADIENTS.json',{'state_label':a.name,'saved_step':saved_step,'saved_next_lr':lr,'batch_scope':'one fixed true main batch and first scheduled8pairs; not sampled actual full training trajectory','auxiliary_frequency':.25,'results':results,'replay':'CUDA foreach AdamW on independent copied tensors and copied states; hypothetical fixed-batch, original model/checkpoint untouched','state_SHA_verified':state is not None,'limitation':'Sparse state sampling cannot prove auxiliary effectiveness over every training step; no efficacy inference from changed coordinates alone'})
    write(out/'DONE.json',{'wall_seconds':time.time()-start,'gpu_hours':(time.time()-start)/3600,'model':a.name,'read_only':True,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'model_sha256':modelhash});print(a.name,'audit complete',time.time()-start)
if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        traceback.print_exc();raise
