from pathlib import Path
import sys,ast,json
import torch
import torch.nn.functional as F
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--upstream',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
cli=parser.parse_args()
from diagnose_official_component_clip import extract
from observe_teacher_clip import make_observer
fn=extract(cli.upstream);fn.decorator_list=[]
ns={'torch':torch,'F':F};exec(compile(ast.Module(body=[fn],type_ignores=[]),'pinned_original_loss','exec'),ns);original=ns['generalized_jsd_loss']
s=torch.tensor([[[.01,.4,.3,.29],[.25,.25,.25,.25]]],dtype=torch.float64).log().requires_grad_(True);t=torch.tensor([[[.9,.05,.03,.02],[.25,.25,.25,.25]]],dtype=torch.float64).log();labels=torch.tensor([[0,-100]])
kwargs=dict(student_logits=s,teacher_logits=t,labels=labels,beta=0,token_clip=.05)
rng=torch.get_rng_state().clone();plain=original(**kwargs);g1,=torch.autograd.grad(plain,s)
records=[];observed=make_observer(original,records,chunk_size=1)(**kwargs);g2,=torch.autograd.grad(observed,s)
assert torch.equal(plain,observed) and torch.equal(g1,g2) and torch.equal(rng,torch.get_rng_state())
assert records[0]['valid_response_positions']==1 and records[0]['masked_response_positions']==1
assert records[0]['teacher_argmax_component_clipped_positions']==1
result={'scope':'CPU invariant check against exact pinned loss, synthetic masked example only','original_loss_equal':True,'original_gradient_equal':True,'torch_rng_unchanged':True,'valid_and_masked_positions_counted_correctly':True,'teacher_argmax_clip_detected':True,'gpu_probe_validated':False}
cli.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
