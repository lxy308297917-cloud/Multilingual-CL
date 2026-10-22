"""All predeclared states and paired objective differences; no winner selection."""
import csv,math
import numpy as np
import torch
import torch.nn.functional as F
from settings import *

def main():
    torch.set_num_threads(4);models=['Base','A','B','C','D'];gradrows=[];identities={};reprs={}
    for name in models+[m+'-mid' for m in 'BCD']:
        folder=RUN/'mechanism'/name;assert (folder/'DONE.json').exists(),name
        identities[name]=json.loads((folder/'IDENTITY.json').read_text())
        if name in models:reprs[name]=json.loads((folder/'REPRESENTATION.json').read_text())
        if name=='A':continue
        obj=json.loads((folder/'GRADIENTS.json').read_text())
        for kind,v in obj['results'].items():
            candidate=v.get('copied_state_AdamW_candidate',{});row={'state':name,'auxiliary_kind':kind,'saved_step':obj['saved_step'],'raw_aux_loss':v['raw_aux_loss'],'main_grad_norm':v['main_grad_L2_fp32_reduction'],'raw_aux_grad_norm':v['aux_grad_L2_fp32_reduction'],'scaled_aux_to_main_ratio':v['scaled_aux_to_main_norm_ratio'],'direction_cosine':v['main_aux_cosine'],'actual_combined_norm_before_clip':v['actual_combined_clip_input_norm_BF16'],'clip_factor':v['combined_clipping_factor'],'changed_preclip_gradient_coordinates':v['aux_changed_preclip_gradient_coordinates'],'changed_candidate_moment_coordinates':candidate.get('moment_changed_coordinates'),'changed_candidate_weight_coordinates':candidate.get('weight_changed_coordinates'),'main_only_candidate_delta_L2':math.sqrt(candidate['main_only_delta_sq']) if candidate else None,'combined_candidate_delta_L2':math.sqrt(candidate['combined_delta_sq']) if candidate else None,'aux_vs_main_candidate_difference_L2':math.sqrt(candidate['difference_delta_sq']) if candidate else None};gradrows.append(row)
    teacher=torch.load(TRAIN/'EN_TEACHER.pt',map_location='cpu',weights_only=False);bank=F.normalize(teacher['vectors'].float(),dim=-1);baseheld=torch.load(TRAIN/'diagnostics/Base/diagnostic_representations.pt',map_location='cpu',mmap=True,weights_only=True)['en']['mean'][10].float();losses={}
    for name in models:
        train=torch.load(RUN/'mechanism'/name/'TRAIN_VECTORS.pt',map_location='cpu',weights_only=True);idx=torch.tensor(train['selected_train_indices']);cache='Base' if name=='Base' else 'align-'+name
        held=torch.load(TRAIN/'diagnostics'/cache/'diagnostic_representations.pt',map_location='cpu',mmap=True,weights_only=True)['sw']['mean'][10].float();losses[name]={}
        for split,sw,en in [('train64',train['sw_vectors'],teacher['vectors'][idx]),('held1000',held,baseheld)]:
            ns=F.normalize(sw.float(),dim=-1);ne=F.normalize(en.float(),dim=-1);pos=(ns*ne).sum(-1);logits=ns@bank.T/.07
            if split=='held1000':logits=torch.cat([pos[:,None]/.07,logits],1);label=torch.zeros(len(sw),dtype=torch.long)
            else:label=idx
            losses[name][split]={'cosine_loss':(1-pos).numpy(),'InfoNCE':F.cross_entropy(logits,label,reduction='none').numpy()}
    objective=[]
    for split,n in [('train64',64),('held1000',1000)]:
        rng=np.random.default_rng(42);boot=rng.integers(0,n,size=(10000,n))
        for a,b in [('A','Base'),('B','A'),('C','A'),('C','B'),('D','A'),('D','B')]:
            for key in ['cosine_loss','InfoNCE']:
                d=losses[a][split][key]-losses[b][split][key];lo,hi=np.quantile(d[boot].mean(1),[.025,.975]);objective.append({'split':split,'comparison':a+'-'+b,'loss':key,'difference_lower_is_better':float(d.mean()),'CI95_low':float(lo),'CI95_high':float(hi),'n':n,'bootstrap':10000,'scope':'fixed training64 or independent diagnostic1000; proxy only'})
    write(REPORT/'MECHANISM_REPRESENTATION.json',reprs);write(REPORT/'MECHANISM_IDENTITY.json',identities);write(REPORT/'MECHANISM_GRADIENTS.json',gradrows);write(REPORT/'OBJECTIVE_PAIRED_CI.json',objective)
    for file,rr in [('MECHANISM_GRADIENTS.csv',gradrows),('OBJECTIVE_PAIRED_CI.csv',objective)]:
        with (REPORT/file).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    print(json.dumps({'states':len(identities),'gradient_probes':len(gradrows),'proxy_comparisons':len(objective)}))
if __name__=='__main__':main()
