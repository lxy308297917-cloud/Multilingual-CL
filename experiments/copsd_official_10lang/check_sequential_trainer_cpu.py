"""Actual Transformers Trainer handoff/reset check on a tiny CPU Qwen3.

The independent reference continues the actual parent model with a new Trainer;
the child instead loads its saved adapter through the production stage helper.
"""
import argparse
import json
from pathlib import Path
import tempfile

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from check_sequential_adapter_cpu import model
from sequential_stage_entry import initialize_stage_model, verify_optimization_start


class VerifyStart(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.proof = verify_optimization_start(self.trainer, state, 0)


def trainer(student, output):
    args = TrainingArguments(output_dir=str(output), use_cpu=True, max_steps=1,
        per_device_train_batch_size=1, gradient_accumulation_steps=1,
        learning_rate=.01, optim='adamw_torch', weight_decay=0, max_grad_norm=0,
        save_steps=1, report_to=[], logging_strategy='no', disable_tqdm=True,
        dataloader_pin_memory=False, seed=123, data_seed=123)
    dataset = [{'input_ids': [1, 3, 8, 5, 2], 'labels': [1, 3, 8, 5, 2],
                'attention_mask': [1, 1, 1, 1, 1]}]
    callback = VerifyStart()
    result = Trainer(model=student, args=args, train_dataset=dataset, callbacks=[callback])
    callback.trainer = result
    return result, callback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(1)
    inputs = torch.tensor([[1, 3, 8, 5, 2]])
    parent_model = model()
    with parent_model.disable_adapter(), torch.no_grad():
        original_teacher = parent_model(inputs).logits.clone()
    with tempfile.TemporaryDirectory(prefix='copsd_stage_trainer_') as directory:
        root = Path(directory)
        parent, parent_start = trainer(parent_model, root / 'parent')
        initialize_stage_model(parent)
        parent.train()
        parent_checkpoint = root / 'parent/checkpoint-1'
        assert parent_checkpoint.is_dir()
        assert {int(x['step']) for x in parent.optimizer.state.values()} == {1}
        with torch.no_grad():
            previous_student = parent.model(inputs).logits.clone()
        assert not torch.equal(previous_student, original_teacher)

        child_model = model()
        child, child_start = trainer(child_model, root / 'child')
        handoff = initialize_stage_model(child, parent_checkpoint, 1)
        with torch.no_grad():
            assert torch.equal(child.model(inputs).logits, previous_student)
        with child.model.disable_adapter(), torch.no_grad():
            assert torch.equal(child.model(inputs).logits, original_teacher)
        child.train()
        assert child_start.proof['optimizer_state_empty']
        assert child_start.proof['scheduler_last_epoch'] == 0
        assert child.optimizer is not parent.optimizer

        # This reference never invokes adapter loading: it already is the
        # trained parent. A new real Trainer supplies independent fresh moments.
        reference, reference_start = trainer(parent.model, root / 'reference')
        reference.train()
        assert reference_start.proof['optimizer_state_empty']
        for (name, left), (other, right) in zip(child.model.named_parameters(), reference.model.named_parameters()):
            assert name == other and torch.equal(left, right), name
        for student in [child.model, reference.model]:
            with student.disable_adapter(), torch.no_grad():
                assert torch.equal(student(inputs).logits, original_teacher)
    handoff.pop('frozen_Base_sha256_PRIVATE')
    result = {'actual_Transformers_Trainer_CPU_used': True,
        'initialization_before_optimizer_creation': True, 'handoff': handoff,
        'actual_parent_optimizer_steps': [1], 'child_train_begin': child_start.proof,
        'independent_reference_train_begin': reference_start.proof,
        'loaded_student_logits_match_parent_before_next_stage': True,
        'child_next_update_equals_existing_parent_with_new_Trainer_bitwise': True,
        'disabled_adapter_teacher_logits_unchanged_through_both_updates': True,
        'scope': 'Tiny synthetic FP32 CPU Trainer initialization/reset integration only. Does not validate actual1.7B GPU rollout, within-stage checkpoint recovery, selected research languages, stage evaluations or effectiveness.'}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
