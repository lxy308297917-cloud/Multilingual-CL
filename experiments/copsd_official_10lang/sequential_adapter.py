"""Adapter-only stage handoff; no Trainer/optimizer/scheduler resume."""
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file


def frozen_fingerprint(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            value = parameter.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def canonical(value):
    return sorted(value) if isinstance(value, (set, list, tuple)) else value


def load_stage_adapter(model, checkpoint, expected_step):
    """Load into the existing default adapter and verify frozen Base unchanged.

    Caller must construct a fresh Trainer optimizer/scheduler and synchronize
    rollout weights afterward. Returned fingerprints are private run evidence.
    """
    checkpoint = Path(checkpoint)
    assert isinstance(model, PeftModel)
    assert set(model.peft_config) == {'default'} and model.active_adapter == 'default'
    assert isinstance(expected_step, int) and expected_step > 0
    assert json.loads((checkpoint / 'trainer_state.json').read_text())['global_step'] == expected_step
    saved_config = json.loads((checkpoint / 'adapter_config.json').read_text())
    current_config = model.peft_config['default'].to_dict()
    fields = ['base_model_name_or_path', 'peft_type', 'task_type', 'r', 'lora_alpha',
              'lora_dropout', 'target_modules', 'bias', 'modules_to_save', 'use_dora',
              'use_rslora', 'rank_pattern', 'alpha_pattern', 'fan_in_fan_out']
    for field in fields:
        assert canonical(saved_config.get(field)) == canonical(current_config.get(field)), field
    assert current_config['bias'] == 'none' and not current_config.get('modules_to_save')
    assert all('lora_' in name for name, p in model.named_parameters() if p.requires_grad)
    trainable = {name: id(p) for name, p in model.named_parameters() if p.requires_grad}
    before = frozen_fingerprint(model)
    weights = load_file(str(checkpoint / 'adapter_model.safetensors'), device='cpu')
    current = get_peft_model_state_dict(model, adapter_name='default')
    assert set(weights) == set(current)
    for name, value in weights.items():
        assert 'lora_' in name and value.shape == current[name].shape
        assert value.dtype == current[name].dtype and torch.isfinite(value).all(), name
    incompatible = set_peft_model_state_dict(model, weights, adapter_name='default')
    assert not incompatible.unexpected_keys
    loaded = get_peft_model_state_dict(model, adapter_name='default')
    assert all(torch.equal(loaded[name].detach().cpu(), value) for name, value in weights.items())
    assert {name: id(p) for name, p in model.named_parameters() if p.requires_grad} == trainable
    assert frozen_fingerprint(model) == before
    return {'previous_global_step': expected_step, 'adapter_count': 1,
            'adapter_values_match_previous_checkpoint': True,
            'parameter_objects_preserved': True, 'frozen_Base_unchanged': True,
            'frozen_Base_sha256_PRIVATE': before,
            'scope': 'Adapter handoff only; caller must separately prove fresh optimizer/scheduler, teacher disabling and current-policy rollout synchronization.'}
