"""Audit terminal 4B resource run; never equate resource success with math benefit."""
import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path
import torch


def read(p):
    return json.loads(Path(p).read_text())


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    code = Path(__file__).parent
    frozen = code/'sw4b_resource2_v3_frozen_20261022.PUBLIC.json'
    f = read(frozen)
    queue = args.root/'gpu-queue/queues/copsd-sw4b-resource2-v3/state.json'
    state = read(queue)
    assert state['phase'] == state['job']['phase'] == 'completed', state['phase']
    assert state['job']['exit_code'] == 0
    assert state['job']['script_sha256'] == f['code_sha256']['run_sw4b_resource2_v3.sh']
    directory = args.root/f['checkpoint_relative']
    proof = read(directory/'COLLOCATE_TRAIN_UPDATES_DONE.json')
    cost = read(directory/'COST_SEGMENT_DONE.json')
    assert proof['global_step'] == proof['completed_updates'] == 2
    assert proof['micro_steps'] == cost['loss_calls'] == 64
    assert proof['base_unchanged_bitwise'] and len(proof['syncs']) == 3
    assert cost['student_forward_calls'] == cost['teacher_forward_calls'] == 64
    assert read(directory/'MASK_REDUCTION_ACTUAL.json')['position_reduction_calls'] == 64
    assert read(directory/'MODEL_SCALE_ACTUAL.json')['all_HF_parameters_on_CUDA']
    assert read(directory/'RUNTIME_SHUTDOWN_DONE.json')['ordered_pool_release_completed']
    prepared = read(code/'sw4b_resource2_prepared_20261022.PUBLIC.json')
    rollouts = read(directory/'RESOURCE_ROLLOUTS.PRIVATE.json')
    assert len(rollouts) == len({r['id'] for r in rollouts}) == 64
    assert set(r['id'] for r in rollouts) == set(prepared['selected_IDs'])
    assert set(prepared['stress_IDs']) <= {r['id'] for r in rollouts}
    assert sum(r['generated_tokens'] for r in rollouts) == cost['valid_response_loss_positions']
    assert max(r['generated_tokens'] for r in rollouts) == 2048
    checkpoint_proofs = []
    for step in [1, 2]:
        cp = directory/f'checkpoint-{step}'
        assert read(cp/'trainer_state.json')['global_step'] == step
        train_args = torch.load(cp/'training_args.bin', map_location='cpu', weights_only=False)
        assert train_args.seed == train_args.data_seed == 42 and train_args.max_steps == 100
        assert train_args.learning_rate == 5e-6 and train_args.gradient_accumulation_steps == 32
        assert torch.load(cp/'scheduler.pt', map_location='cpu', weights_only=False)['last_epoch'] == step
        optimizer = torch.load(cp/'optimizer.pt', map_location='cpu', weights_only=False)
        assert {int(s['step']) for s in optimizer['state'].values()} == {step}
        assert (cp/'rng_state.pth').stat().st_size and (cp/'adapter_model.safetensors').stat().st_size
        checkpoint_proofs.append({'step': step, 'optimizer_scheduler_RNG_present': True,
            'adapter_sha256': digest(cp/'adapter_model.safetensors')})
    telemetry = args.root/'logs/copsd_official_10lang/sw4b_resource2_20261022_v3.gpu.csv'
    samples = [line.split(',') for line in telemetry.read_text().splitlines() if line.strip()]
    peak = max(float(row[1]) for row in samples)
    # Physical 3090 capacity from nvidia-smi, not misleading sleeping-pool accounting.
    total_mib = float(subprocess.check_output(['nvidia-smi', '-i', state['job']['gpu_uuids'][0], '--query-gpu=memory.total', '--format=csv,noheader,nounits'], text=True).strip())
    assert peak < total_mib
    start = datetime.datetime.fromisoformat(state['job']['started_at'])
    finish = datetime.datetime.fromisoformat(state['job']['finished_at'])
    result = {'protocol': 'SW4B_resource2_v3_full2048_completion_audit',
        'completed_at': state['job']['finished_at'], 'engineering_completed': True,
        'model_revision': f['model_revision'], 'frozen_sha256': digest(frozen),
        'core_original_source_sha256': read(code/'mask_reduction_memory_CUDA_20261022.PUBLIC.json')['original_source_sha256'],
        'mathematical_loss_retained_with_reduction_order_change': True,
        'CUDA_fixture_loss_and_student_gradient_bitwise_equal': True,
        'completed_updates': 2, 'actual_exposures': 64, 'teacher_CPU_offload': False,
        'frozen_Base_bitwise_unchanged': True, 'rollout_sync_count': 3,
        'generated_tokens': sum(r['generated_tokens'] for r in rollouts),
        'responses_at2048': sum(r['generated_tokens'] == 2048 for r in rollouts),
        'valid_loss_positions': cost['valid_response_loss_positions'],
        'masked_loss_positions': cost['masked_response_positions'],
        'max_teacher_prompt_tokens': max(r['teacher_prompt_tokens'] for r in rollouts),
        'max_student_prompt_tokens': max(r['student_prompt_tokens'] for r in rollouts),
        'training_seconds_including_diagnostic_hashing': proof['train_seconds_including_diagnostic_hashing'],
        'physical_peak_MiB': peak, 'sampled_physical_headroom_MiB': total_mib-peak,
        'telemetry_interval_seconds': 1, 'telemetry_sha256': digest(telemetry),
        'allocated_parent_GPU_hours': (finish-start).total_seconds()/3600,
        'checkpoint_proofs': checkpoint_proofs,
        'checkpoint_actual_reload_tested': False,
        'mathematical_benefit_tested': False,
        'scope': '64 engineering trajectories only; not efficacy, convergence, a100-update reproduction, nor validation of future unobserved contexts. Debug weights cannot initialize the future fresh pilot.'}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
