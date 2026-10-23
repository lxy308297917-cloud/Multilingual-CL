"""Clone only response-logit views so discarded prompt logits can be freed.

Exactly two storage changes in pinned compute_loss; dtype, values, causal slice,
mask, teacher context, vocabulary, clipping and loss arithmetic stay unchanged.
Original core source remains intact. Never use contiguous(): these views may
already be contiguous while retaining the full prompt-plus-response storage.
"""
import ast
import inspect
import textwrap


def make_compact_compute_loss(original):
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    return transform_compute_loss(tree,original.__globals__)


def transform_compute_loss(tree,namespace):
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_loss')
    changed=[]
    for node in ast.walk(fn):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            target=node.targets[0].id
            if target in ['student_logits_for_loss','teacher_logits_for_loss']:
                expected='student_logits' if target.startswith('student') else 'teacher_logits'
                assert isinstance(node.value,ast.Name) and node.value.id==expected
                node.value=ast.Call(func=ast.Attribute(value=node.value,attr='clone',ctx=ast.Load()),args=[],keywords=[]);changed.append(target)
    assert sorted(changed)==['student_logits_for_loss','teacher_logits_for_loss']
    fn.decorator_list=[];tree=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));local={}
    exec(compile(tree,'response_logit_storage_compaction','exec'),namespace,local)
    return local['compute_loss']
