"""Export an independent BF16 evaluation-only Base+LoRA merge on CPU.

Keeps source Base/adapter files untouched. A rounded dense merge is not
bitwise equivalent to native LoRA, so this is an engineering export, not a
replacement for existing scores or an official-training modification.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b''):
            value.update(block)
    return value.hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    a = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only export required'
    assert not a.output.exists() and not a.report.exists()
    assert a.output.resolve().is_relative_to((a.root / 'models/checkpoint/copsd_eval_dense_exports_v1').resolve())
    assert a.adapter.resolve().is_relative_to((a.root / 'models/checkpoint').resolve())
    assert shutil.disk_usage(a.root).free >= 25 * 2**30
    code = Path(__file__).parent
    lock = read(a.root / 'logs/copsd_official_10lang/eval_vllm_runtime_20261019.CONFIG.json')
    assert digest(lock['model_lock']['path']) == lock['model_lock']['sha256']
    model_lock = read(lock['model_lock']['path'])
    base_path = a.root / 'models/Qwen3-1.7B'
    assert model_lock['repository'] == 'Qwen/Qwen3-1.7B'
    assert model_lock['revision'] == '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
    tick = time.monotonic()
    metadata = {}
    for name, item in model_lock['files'].items():
        source = base_path / name
        stat = source.stat()
        assert stat.st_size == item['bytes'] and digest(source) == item['sha256']
        metadata[name] = (stat.st_size, stat.st_mtime_ns)
    base_verified_seconds = time.monotonic() - tick
    adapter_sha = digest(a.adapter / 'adapter_model.safetensors')
    adapter_config_sha = digest(a.adapter / 'adapter_config.json')
    assert read(a.adapter / 'trainer_state.json')['global_step'] == 20
    config = read(a.adapter / 'adapter_config.json')
    assert config['r'] == 64 and config['lora_alpha'] == 128 and config['bias'] == 'none'
    assert not config.get('modules_to_save') and not config.get('use_dora')
    reference_config = a.root / 'logs/copsd_official_10lang/development_vllm8192/sw_clip20_dev100_8192_20261020_v1/CONFIG.json'
    # The reference path comes from the frozen native-LoRA performance plan.
    native_lock = read(code / 'eval_concurrency8192_frozen_20261021.PUBLIC.json')
    reference_config = a.root / native_lock['reference_config_relative']
    assert digest(reference_config) == native_lock['reference_config_sha256']
    reference = read(reference_config)
    assert reference['adapter'] == str(a.adapter) and reference['language'] == 'sw'
    assert reference['adapter'] and digest(reference['input']) == reference['input_sha256']
    rows = read(reference['input'])[:64]
    assert len(rows) == 64
    a.output.mkdir(parents=True)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from evaluate_vllm import official_api
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    api = official_api(Path(reference['source']))
    prompts = [api['build_qwen3_prompt'](tokenizer, row['problem_sw'], 'SW', True) for row in rows]
    original_ids = tokenizer(prompts, add_special_tokens=False)['input_ids']
    tick = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True, low_cpu_mem_usage=True)
    student = PeftModel.from_pretrained(model, a.adapter, is_trainable=False).eval()
    assert all(p.device.type == 'cpu' for p in student.parameters())
    assert set(student.peft_config) == {'default'} and not torch.cuda.is_initialized()
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    probe = tokenizer(prompts[:2], return_tensors='pt', padding=True, add_special_tokens=False)
    assert max(probe['input_ids'].shape) < 4096
    with torch.inference_mode():
        before = student(**probe, logits_to_keep=1, use_cache=False).logits[:, -1].float()
        merged = student.merge_and_unload(safe_merge=True).eval()
        after = merged(**probe, logits_to_keep=1, use_cache=False).logits[:, -1].float()
    assert torch.isfinite(before).all() and torch.isfinite(after).all()
    assert not any('lora_' in name for name, _ in merged.named_parameters())
    assert all(p.dtype == torch.bfloat16 for p in merged.parameters())
    log_before, log_after = before.log_softmax(-1), after.log_softmax(-1)
    numerical = {'probe_prompts': 2, 'probe_ids': [row['problem_id'] for row in rows[:2]],
        'max_absolute_last_position_logit_difference': float((before - after).abs().max()),
        'mean_absolute_last_position_logit_difference': float((before - after).abs().mean()),
        'KL_native_CPU_to_merged_CPU_last_position': (log_before.exp() * (log_before - log_after)).sum(-1).tolist(),
        'last_position_top1_matches': (before.argmax(-1) == after.argmax(-1)).tolist(),
        'scope': 'Two real SW prompts on CPU only; not native-vLLM equivalence, all-token fidelity, generation equivalence or mathematics accuracy.'}
    tokenizer.padding_side = old_padding_side
    merged.save_pretrained(a.output, safe_serialization=True, max_shard_size='4GB')
    tokenizer.save_pretrained(a.output)
    exported_tokenizer = AutoTokenizer.from_pretrained(a.output, local_files_only=True)
    assert exported_tokenizer(prompts, add_special_tokens=False)['input_ids'] == original_ids
    assert [api['build_qwen3_prompt'](exported_tokenizer, row['problem_sw'], 'SW', True) for row in rows] == prompts
    assert not torch.cuda.is_initialized()
    for name, value in metadata.items():
        source = base_path / name
        assert (source.stat().st_size, source.stat().st_mtime_ns) == value
    assert digest(a.adapter / 'adapter_model.safetensors') == adapter_sha
    assert digest(a.adapter / 'adapter_config.json') == adapter_config_sha
    files = {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in a.output.iterdir() if p.is_file()}
    private = {'completed': True, 'backend': 'CPU_PEFT_safe_merge_BF16_v1',
        'model_revision': model_lock['revision'], 'source_adapter': str(a.adapter),
        'source_adapter_sha256': adapter_sha, 'source_adapter_config_sha256': adapter_config_sha,
        'model_lock_sha256': lock['model_lock']['sha256'], 'exporter_sha256': digest(Path(__file__)),
        'files': files, 'numerical_probe': numerical,
        'reference_native_benchmark_config_sha256': digest(reference_config)}
    write_new(a.output / 'EXPORT.MANIFEST.PRIVATE.json', private)
    public = {'observed_at': datetime.datetime.now().isoformat(), 'model': 'Qwen3-1.7B',
        'training_language': 'sw', 'source_checkpoint_step': 20, 'source_training_seed': 42,
        'backend': private['backend'], 'model_revision': model_lock['revision'],
        'source_adapter_sha256': adapter_sha, 'exporter_sha256': private['exporter_sha256'],
        'export_manifest_sha256': digest(a.output / 'EXPORT.MANIFEST.PRIVATE.json'),
        'export_files': files, 'base_source_files_stream_SHA256_verified': len(metadata),
        'source_Base_metadata_and_adapter_files_unchanged': True,
        'native_and_exported_tokenizer_prompts_and_ids_match_for64_frozen_SW_prompts': True,
        'numerical_probe': numerical, 'source_verification_seconds': base_verified_seconds,
        'load_merge_probe_save_seconds': time.monotonic() - tick,
        'GPU_used': False, 'new_training': False, 'new_math_evaluation': False,
        'scope': 'Independent evaluation-only BF16 merge for throughput exploration. CPU last-token checks do not establish native-LoRA sampling equivalence or accuracy preservation. Current native-LoRA formal outputs/checkpoints stay unchanged; any adopted dense backend requires a separate fixed evaluation protocol for Base and all comparison models.'}
    write_new(a.report, public)
    print(json.dumps({'CPU_export_completed': True, 'files': len(files), 'numerical_probe': numerical}))


if __name__ == '__main__':
    main()
