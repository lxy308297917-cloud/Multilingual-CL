"""Answer-position alignment and full-vocabulary reverse KL for SW OPD."""
import torch


def aligned_reverse_kl(student_logits, teacher_logits, student_prompt_tokens, teacher_prompt_tokens, answer_mask, chunk_tokens=16):
    """KL(student || frozen teacher) on the same sampled answer prefix.

    The logits at prompt length - 1 predict the first answer token. Inputs are
    unshifted causal-model logits for prompt + identical sampled answer IDs.
    answer_mask includes real generated tokens, including a real EOS, and
    excludes padding. Teacher weights must be frozen by the caller.
    """
    if student_logits.ndim != 3 or teacher_logits.ndim != 3:
        raise ValueError("Expected [batch, sequence, vocabulary] logits")
    if student_logits.shape[0] != teacher_logits.shape[0] or student_logits.shape[2] != teacher_logits.shape[2]:
        raise ValueError("Student and teacher batch/vocabulary differ")
    if answer_mask.ndim != 2 or answer_mask.shape[0] != student_logits.shape[0]:
        raise ValueError("Answer mask shape mismatch")
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive")
    batch, answer_width = answer_mask.shape
    if not 1 <= student_prompt_tokens or not 1 <= teacher_prompt_tokens:
        raise ValueError("Prompt lengths must be positive")
    if student_prompt_tokens - 1 + answer_width > student_logits.shape[1]:
        raise ValueError("Student logits do not cover answer predictions")
    if teacher_prompt_tokens - 1 + answer_width > teacher_logits.shape[1]:
        raise ValueError("Teacher logits do not cover answer predictions")
    if answer_mask.sum() == 0:
        raise ValueError("No effective answer tokens")
    total = student_logits.new_zeros((), dtype=torch.float32)
    count = answer_mask.float().sum()
    for start in range(0, answer_width, chunk_tokens):
        stop = min(start + chunk_tokens, answer_width)
        s = student_logits[:, student_prompt_tokens - 1 + start:student_prompt_tokens - 1 + stop].float()
        t = teacher_logits[:, teacher_prompt_tokens - 1 + start:teacher_prompt_tokens - 1 + stop].detach().float()
        logp = torch.log_softmax(s, dim=-1)
        logq = torch.log_softmax(t, dim=-1)
        per_token = (logp.exp() * (logp - logq)).sum(dim=-1)
        total = total + (per_token * answer_mask[:, start:stop].float()).sum()
    return total / count
