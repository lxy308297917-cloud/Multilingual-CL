"""CPU algorithm check; this is not a Qwen3 OPD training result."""
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from sw_opd_core import aligned_reverse_kl, aligned_reverse_kl_tail


class TinyCausalLM(nn.Module):
    def __init__(self, vocab=32, hidden=24):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, ids):
        hidden, _ = self.rnn(self.embed(ids))
        return self.head(hidden)


def digest(model):
    return hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()


def sample_current_student(model, prompt, steps, generator):
    ids = prompt.clone()
    with torch.no_grad():
        for _ in range(steps):
            logits = model(ids)[:, -1] / 1.1
            token = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator)
            ids = torch.cat((ids, token), dim=1)
    return ids[:, prompt.shape[1]:]


def main():
    torch.manual_seed(42)
    student = TinyCausalLM()
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    teacher_before = digest(teacher)
    student_before = digest(student)
    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-3)
    student_prompt = torch.tensor([[2, 4, 5]], dtype=torch.long)
    teacher_prompt = torch.tensor([[2, 4, 5, 6, 7]], dtype=torch.long)
    records = []
    for step in range(2):
        generator = torch.Generator().manual_seed(100 + step)
        answer = sample_current_student(student, student_prompt, 6, generator)
        s_all = torch.cat((student_prompt, answer), dim=1)
        t_all = torch.cat((teacher_prompt, answer), dim=1)
        s_logits = student(s_all)
        with torch.no_grad():
            t_logits = teacher(t_all)
        mask = torch.ones_like(answer, dtype=torch.bool)
        loss = aligned_reverse_kl(s_logits, t_logits, student_prompt.shape[1], teacher_prompt.shape[1], mask, chunk_tokens=2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.sqrt(sum(p.grad.square().sum() for p in student.parameters() if p.grad is not None))
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm) or grad_norm <= 0:
            raise RuntimeError("Invalid student KL/gradient")
        if any(p.grad is not None for p in teacher.parameters()):
            raise RuntimeError("Teacher acquired gradient")
        optimizer.step()
        records.append({"policy_version": step, "sampled_answer_ids": answer[0].tolist(), "effective_loss_tokens": int(mask.sum()), "reverse_kl": float(loss), "student_grad_norm": float(grad_norm)})
    if digest(teacher) != teacher_before or digest(student) == student_before:
        raise RuntimeError("Teacher changed or student failed to update")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    tiny_qwen = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64, attention_dropout=0.0)).eval()
    answer = torch.tensor([[8, 9, 10, 11]])
    qwen_mask = torch.ones_like(answer, dtype=torch.bool)
    s_full_ids = torch.cat((student_prompt, answer), dim=1)
    t_full_ids = torch.cat((teacher_prompt, answer), dim=1)
    with torch.no_grad():
        s_full = tiny_qwen(s_full_ids).logits
        t_full = tiny_qwen(t_full_ids).logits
        s_tail = tiny_qwen(s_full_ids[:, :-1], logits_to_keep=answer.shape[1]).logits
        t_tail = tiny_qwen(t_full_ids[:, :-1], logits_to_keep=answer.shape[1]).logits
        old_kl = aligned_reverse_kl(s_full, t_full, student_prompt.shape[1], teacher_prompt.shape[1], qwen_mask, chunk_tokens=2)
        tail_kl = aligned_reverse_kl_tail(s_tail, t_tail, qwen_mask, chunk_tokens=2)
    if not torch.allclose(old_kl, tail_kl, atol=1e-6, rtol=1e-5):
        raise RuntimeError(f"Qwen3 tail/full causal alignment mismatch: {old_kl} vs {tail_kl}")
    result = {"status": "passed", "scope": "CPU tiny-model algorithm and randomly initialized tiny Qwen3 alignment only; not a 4B/FSDP training smoke", "updates": 2, "teacher_hash_unchanged": True, "student_hash_changed": True, "qwen3_tail_full_kl_match": True, "qwen3_reverse_kl": float(tail_kl), "records": records}
    out = Path('/data0/users/lixiaoyang/logs/math_sw_opd_20261015/SW_OPD_ALGORITHM_SMOKE.json')
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
