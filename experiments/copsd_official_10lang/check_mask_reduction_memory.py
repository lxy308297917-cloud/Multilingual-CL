"""Check scalar and student-gradient equivalence against pinned original loss."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from diagnose_official_component_clip import extract
from mask_reduction_memory import make_mask_reduction_loss


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    fn=extract(a.upstream);fn.decorator_list=[];ns={'torch':torch,'F':F};exec(compile(ast.Module(body=[fn],type_ignores=[]),'pinned_original_loss','exec'),ns);original=ns['generalized_jsd_loss'];counts={};fast=make_mask_reduction_loss(original,counts)
    torch.manual_seed(20261022)
    types=[torch.float64,torch.float32,torch.bfloat16] if a.device=='cpu' else [torch.float32,torch.bfloat16]
    shape=(2,3,17) if a.device=='cpu' else (1,8,151936)
    cases=[]
    for dtype in types:
        for partial in [False,True]:
            data=torch.randn(shape,device=a.device,dtype=dtype);teacher=torch.randn(shape,device=a.device,dtype=dtype)
            data[...,0]=-4;teacher[...,0]=12
            labels=torch.zeros(shape[:2],device=a.device,dtype=torch.long)
            if partial:labels[0,-1]=-100
            first=data.clone().requires_grad_(True);second=data.clone().requires_grad_(True)
            kwargs=dict(teacher_logits=teacher,labels=labels,beta=0,temperature=1.1,token_clip=.05,top_k=None,reduction='batchmean')
            autocast=lambda:torch.autocast(device_type=a.device,dtype=torch.bfloat16,enabled=a.device=='cuda')
            with autocast():one=original(student_logits=first,**kwargs)
            g1,=torch.autograd.grad(one,first)
            with autocast():two=fast(student_logits=second,**kwargs)
            g2,=torch.autograd.grad(two,second)
            loss_diff=float((one-two).abs().item());grad_diff=float((g1-g2).abs().max().item())
            assert torch.isfinite(one) and torch.isfinite(two)
            assert torch.allclose(one,two,atol=1e-6,rtol=1e-6) and torch.allclose(g1,g2,atol=1e-6,rtol=1e-6),(dtype,partial,loss_diff,grad_diff)
            cases.append({'dtype':str(dtype),'partial_mask':partial,'loss_equal_bitwise':torch.equal(one,two),'gradient_equal_bitwise':torch.equal(g1,g2),'loss_max_abs_difference':loss_diff,'gradient_max_abs_difference':grad_diff})
    out={'device':a.device,'shape':list(shape),'cases':cases,'counts':counts,'all_passed':True,'original_source_sha256':sha(a.upstream),'wrapper_sha256':sha(Path(__file__).with_name('mask_reduction_memory.py')),
        'scope':('CPU fixture only, noGPU validation yet. ' if a.device=='cpu' else 'CUDA fullvocabulary151936 withbf16autocast. ')+ 'Pinned beta0/temperature1.1/clip0.05 loss and studentgradient equivalence, not4B memory or downstream efficacy. Original vocabulary components retained; sum vocabulary before position selection. Floating-point reduction order may differ.'}
    with a.output.open('x') as z:json.dump(out,z,indent=2);z.write('\n')
    print(json.dumps(out))


if __name__=='__main__':main()
