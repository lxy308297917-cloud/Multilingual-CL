"""CPU sensitivity example for the exact pinned official loss, not model training.

Shows what component clipping can do; does not establish the cause of any real
run's regression. Does not patch the trainer, alter optimizers or launch GPUs.
"""
import argparse
import ast
import json
from pathlib import Path
import torch
import torch.nn.functional as F


def extract(path):
    module=ast.parse(path.read_text())
    klass=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='OPSDTrainer')
    return next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='generalized_jsd_loss')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    fn=extract(args.source);original=extract(args.upstream)
    assert ast.dump(fn,include_attributes=False)==ast.dump(original,include_attributes=False)
    fn.decorator_list=[]
    namespace={'torch':torch,'F':F}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(args.source),'exec'),namespace)
    loss_fn=namespace['generalized_jsd_loss']
    teacher=torch.tensor([.90,.05,.03,.02],dtype=torch.float64).reshape(1,1,-1)
    student=torch.tensor([.01,.40,.30,.29],dtype=torch.float64).reshape(1,1,-1)
    results=[]
    for clip in [None,.05]:
        logits=student.log().detach().requires_grad_(True)
        loss=loss_fn(logits,teacher.log(),labels=torch.tensor([[0]]),beta=0,
                     temperature=1.0,top_k=None,token_clip=clip)
        gradient,=torch.autograd.grad(loss,logits)
        results.append({'component_clip':clip,'loss':float(loss.detach()),
                        'student_logit_gradients':gradient.flatten().tolist(),
                        'teacher_top_token_logit_gradient':float(gradient.flatten()[0]),
                        'descent_effect_on_teacher_top_token_logit':'increase' if gradient.flatten()[0]<0 else 'decrease'})
    assert results[0]['teacher_top_token_logit_gradient']<0
    assert results[1]['teacher_top_token_logit_gradient']>0
    report={'scope':'Synthetic CPU autograd counterexample, not real-model evidence or a proposed training replacement',
            'official_commit':'4f4410730e9b6775a30d20f4b6e141cba2e87590',
            'adapted_loss_AST_equal_pinned_upstream':True,'beta':0,
            'teacher_distribution':teacher.flatten().tolist(),
            'student_distribution':student.flatten().tolist(),'results':results,
            'finding':'The released code clips individual vocabulary contributions before vocabulary reduction. This can suppress the teacher-top-token term and reverse its student-logit gradient sign.',
            'limits':['Does not prove that this pattern is frequent in actual trajectories or explains measured regressions.',
                      'No running code or scientific configuration changed. Current official-reference runs retain clip0.05.',
                      'Any future unclipped-loss comparison must be a separately labelled project ablation with matched data/budget/evaluation.']}
    if args.output.exists():raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
