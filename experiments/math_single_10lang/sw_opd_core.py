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


def aligned_reverse_kl_tail(student_tail_logits, teacher_tail_logits, answer_mask, chunk_tokens=16):
    """Full-vocabulary KL when each model returns only answer-prediction logits.

    Feed each model `prompt_ids + answer_ids[:-1]` with Qwen3's
    `logits_to_keep=len(answer_ids)`. The resulting tail starts at the logit
    that predicts answer_ids[0], irrespective of the two prompt lengths.
    This avoids materializing logits for the privileged teacher prompt.
    """
    if student_tail_logits.shape != teacher_tail_logits.shape or student_tail_logits.ndim != 3:
        raise ValueError("Student and teacher answer logits must have the same [batch, tokens, vocab] shape")
    if answer_mask.shape != student_tail_logits.shape[:2]:
        raise ValueError("Answer mask must match returned answer positions")
    if chunk_tokens < 1 or not bool(answer_mask.any()):
        raise ValueError("Need positive chunk size and at least one effective answer token")
    total = student_tail_logits.new_zeros((), dtype=torch.float32)
    count = answer_mask.float().sum()
    for start in range(0, answer_mask.shape[1], chunk_tokens):
        stop = min(start + chunk_tokens, answer_mask.shape[1])
        s = student_tail_logits[:, start:stop].float()
        t = teacher_tail_logits[:, start:stop].detach().float()
        logp = torch.log_softmax(s, dim=-1)
        logq = torch.log_softmax(t, dim=-1)
        total = total + ((logp.exp() * (logp - logq)).sum(dim=-1) * answer_mask[:, start:stop].float()).sum()
    return total / count
