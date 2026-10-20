"""CPU handoff check on a tiny synthetic Qwen3, not a mathematics experiment."""
import argparse
import json
from pathlib import Path
import tempfile

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from sequential_adapter import load_stage_adapter


def model():
    torch.manual_seed(77)
    base = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=32,
        attention_dropout=0.0, tie_word_embeddings=False))
    return get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
        target_modules=['q_proj'], bias='none', task_type='CAUSAL_LM')).eval()


def optimizer(m):
    return torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=0.01)


def update(m, opt, inputs):
    opt.zero_grad(set_to_none=True)
    m(inputs, labels=inputs).loss.backward()
    opt.step()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(1)
    first = model()
    inputs = torch.tensor([[1, 3, 8, 5, 2]])
    with first.disable_adapter(), torch.no_grad():
        teacher_before = first(inputs).logits.clone()
    previous_optimizer = optimizer(first)
    update(first, previous_optimizer, inputs)
    with torch.no_grad():
        previous_student = first(inputs).logits.clone()
    assert not torch.equal(previous_student, teacher_before)
    with tempfile.TemporaryDirectory(prefix='copsd_adapter_handoff_') as tmp:
        path = Path(tmp)
        first.save_pretrained(path)
        (path / 'trainer_state.json').write_text(json.dumps({'global_step': 1}))
        second = model()
        proof = load_stage_adapter(second, path, expected_step=1)
        with second.disable_adapter(), torch.no_grad():
            assert torch.equal(second(inputs).logits, teacher_before)
        with torch.no_grad():
            assert torch.equal(second(inputs).logits, previous_student)
        fresh = optimizer(second)
        scheduler = torch.optim.lr_scheduler.LambdaLR(fresh, lambda _: 1.0)
        assert not fresh.state and scheduler.last_epoch == 0
        assert {int(v['step']) for v in previous_optimizer.state.values()} == {1}
        # Independent reference also resets its optimizer, then consumes the same
        # next batch. This catches a hidden optimizer-state resume at handoff.
        update(first, optimizer(first), inputs)
        update(second, fresh, inputs)
        for (name, a), (other, b) in zip(first.named_parameters(), second.named_parameters()):
            assert name == other and torch.equal(a, b), name
        with second.disable_adapter(), torch.no_grad():
            assert torch.equal(second(inputs).logits, teacher_before)
    proof.pop('frozen_Base_sha256_PRIVATE')
    proof.update(tiny_Qwen3_CPU_only=True, teacher_logits_unchanged=True,
        loaded_student_logits_match_previous=True, fresh_optimizer_empty=True,
        fresh_scheduler_last_epoch=0, next_update_matches_reset_reference=True,
        scope='Synthetic CPU handoff only. No actual1.7B stage training, GPU rollout synchronization, recovery or method effectiveness established.')
    with args.output.open('x') as stream:
        json.dump(proof, stream, indent=2)
        stream.write('\n')
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
