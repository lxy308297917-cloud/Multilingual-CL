"""Observe real official loss inputs without changing the returned loss/gradient.

Chunked, detached statistics only. Does not replace KL, masks or optimization.
"""
import inspect
import torch
import torch.nn.functional as F


def make_observer(original, records, chunk_size=32):
    signature=inspect.signature(original)
    def observed(*args, **kwargs):
        bound=signature.bind(*args,**kwargs);bound.apply_defaults();values=bound.arguments
        assert values['beta']==0 and values['top_k'] is None
        assert values['token_clip']==.05 and not values['logits_are_probs']
        student=values['student_logits'];teacher=values['teacher_logits'];labels=values['labels']
        assert labels is not None and student.shape==teacher.shape and student.shape[:2]==labels.shape
        row={'valid_response_positions':0,'masked_response_positions':int((labels==-100).sum()),
             'teacher_argmax_component_clipped_positions':0,'argmax_disagreement_positions':0,
             'teacher_probability_mass_sum':0.,'clipped_teacher_probability_mass_sum':0.,
             'unclipped_component_sum':0.,'clipped_component_sum':0.,
             'teacher_minus_student_sampled_logprob_sum':0.}
        # Match the original log-softmax/component dtype before reduction.
        # Float32 sums are diagnostics only; the original loss is called below.
        with torch.no_grad():
            for start in range(0,labels.shape[1],chunk_size):
                end=start+chunk_size;valid=labels[:,start:end]!=-100
                if not bool(valid.any()):continue
                sp=F.log_softmax(student[:,start:end].detach()/values['temperature'],dim=-1)
                tp=F.log_softmax(teacher[:,start:end].detach()/values['temperature'],dim=-1)
                components=F.kl_div(sp,tp,reduction='none',log_target=True)
                clipped=components>.05
                probability=tp.float().exp()
                top=tp.argmax(-1,keepdim=True)
                sample_ids=labels[:,start:end].clamp_min(0).unsqueeze(-1)
                row['valid_response_positions']+=int(valid.sum())
                row['teacher_argmax_component_clipped_positions']+=int(clipped.gather(-1,top).squeeze(-1)[valid].sum())
                row['argmax_disagreement_positions']+=int((sp.argmax(-1)!=top.squeeze(-1))[valid].sum())
                row['teacher_probability_mass_sum']+=float(probability.sum(-1)[valid].sum())
                row['clipped_teacher_probability_mass_sum']+=float((probability*clipped).sum(-1)[valid].sum())
                row['unclipped_component_sum']+=float(components.float().sum(-1)[valid].sum())
                row['clipped_component_sum']+=float(components.clamp(max=.05).float().sum(-1)[valid].sum())
                row['teacher_minus_student_sampled_logprob_sum']+=float((tp.gather(-1,sample_ids)-sp.gather(-1,sample_ids)).squeeze(-1)[valid].float().sum())
                del sp,tp,components,clipped,probability,top,sample_ids
        records.append(row)
        return original(*args,**kwargs)
    return observed
