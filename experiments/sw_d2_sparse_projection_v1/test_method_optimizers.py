import copy
import torch
from method_optimizers import ElementwiseProtectedAdamW,ProjectedRowAdamW

def run():
    torch.manual_seed(42)
    p=torch.nn.Parameter(torch.randn(8,8,dtype=torch.float32));initial=p.detach().clone()
    def source(_):return {'w':torch.ones_like(p).cpu()},{'examples':32}
    opt=ElementwiseProtectedAdamW({'w':p},source_provider=source,protected=True,fraction=.25,mask_interval=1,source_interval=2)
    for step in range(3):
        before=p.detach().clone();p.grad=torch.randn_like(p);opt.param_groups[0]['lr']=5e-5
        opt.step();mask=opt.masks['w'];assert int(mask.sum())==16
        assert torch.equal(p[~mask],before[~mask]);assert bool((p[mask]!=before[mask]).any())
    saved=copy.deepcopy(opt.state_dict());q=torch.nn.Parameter(p.detach().clone())
    other=ElementwiseProtectedAdamW({'w':q},source_provider=lambda _:({'w':torch.ones_like(q).cpu()},{}),protected=True,fraction=.25,mask_interval=1,source_interval=2)
    other.load_state_dict(saved)
    assert other.steps==opt.steps
    assert torch.equal(other.state[q]['m'],opt.state[p]['m'])
    assert torch.equal(other.masks['w'],opt.masks['w'])
    u=torch.zeros(8,2);u[:2,:]=torch.eye(2)
    r=torch.nn.Parameter(torch.randn(8,8));start=r.detach().clone()
    proj=ProjectedRowAdamW({'w':r},{'w':u},alpha=1.,row_fraction=.5,mask_interval=1)
    r.grad=torch.randn_like(r);proj.step();rows=proj.masks['w'];assert int(rows.sum())==4
    assert torch.equal(r[~rows],start[~rows])
    # FP32 reference update: projection is exact before BF16 rounding.
    d=(r-start);assert torch.allclose(d@u,torch.zeros(8,2),atol=1e-6)
    save=copy.deepcopy(proj.state_dict());s=torch.nn.Parameter(r.detach().clone())
    restored=ProjectedRowAdamW({'w':s},{'w':u},alpha=1.,row_fraction=.5,mask_interval=1)
    restored.load_state_dict(save);assert restored.steps==1
    if torch.cuda.is_available():
        moved=copy.deepcopy(save);moved['mechanism']['bases']['w']=moved['mechanism']['bases']['w'].cuda()
        reloaded=ProjectedRowAdamW({'w':torch.nn.Parameter(s.detach().clone())},{'w':u},alpha=1.,row_fraction=.5,mask_interval=1)
        reloaded.load_state_dict(moved);assert reloaded.steps==1
    assert torch.equal(restored.masks['w'],rows)
    print('PASS: masks, exact inactive coordinates, projection, FP32 moments, optimizer resume')
if __name__=='__main__':run()
