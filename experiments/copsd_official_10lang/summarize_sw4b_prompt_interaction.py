"""Accept all four fixed SW4B model x thinking-prefix cells; no new scoring."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    code = Path(__file__).parent
    freeze = read(code/'sw4b_trained_prefix_frozen_20261022.PUBLIC.json')
    state = read(args.root/'gpu-queue/queues/copsd-sw4b-trained-prefix-v1/state.json')
    assert state['phase'] == 'completed' and state['job']['exit_code'] == 0
    assert state['job']['script_sha256'] == freeze['code_sha256']['run_sw4b_trained_prefix_diagnostic.sh']
    for name, digest in freeze['code_sha256'].items():
        assert sha(code/name) == digest, name
    output_root = args.root/'logs/copsd_official_10lang/development_vllm8192'
    ids = {'Base_SWprefix': freeze['native_Base_run_id'],
           'Base_ENprefix': read(code/'sw4b_prefix_only_frozen_20261022.PUBLIC.json')['run_id'],
           'SW20_ENprefix': freeze['run_id'],
           'SW20_SWprefix': 'sw_4b_server_pilot20_safe2700_horizon100_seed42_20261022_v1_sw_dev8192'}
    cases, rows, configs = {}, {}, {}
    keys = ['model_revision', 'language', 'input_sha256', 'official_source_sha256',
            'released_length_script_sha256', 'items', 'samples', 'max_new_tokens',
            'temperature', 'top_p', 'top_k', 'seed', 'max_num_seqs', 'packages']
    for condition, run_id in ids.items():
        folder = output_root/run_id
        config, done = read(folder/'CONFIG.json'), read(folder/'DONE.json')
        data = [json.loads(line) for line in (folder/'ITEMS.jsonl').read_text().splitlines()]
        assert len(data) == len({str(r['id']) for r in data}) == 100
        assert all(len(r['samples']) == 1 for r in data)
        samples = [r['samples'][0] for r in data]
        measured = {'single_correct': sum(s['correct'] for s in samples),
                    'generated_tokens': sum(s['tokens'] for s in samples),
                    'normal_terminations': sum(s['finish_reason'] == 'stop' for s in samples),
                    'truncated': sum(s['finish_reason'] == 'length' for s in samples),
                    'answer_extraction_failures': sum(s['boxed_answer'] is None for s in samples)}
        assert all(done[k] == v for k, v in measured.items())
        assert all(s['tokens'] == len(s['token_ids']) <= 8192 for s in samples)
        assert done['items'] == 100 and done['n'] == 1 and not done['formal_test']
        cases[condition] = {**measured, 'items': 100, 'generation_seconds': done['generation_seconds'],
                            'ITEMS_sha256': sha(folder/'ITEMS.jsonl'), 'CONFIG_sha256': sha(folder/'CONFIG.json'),
                            'DONE_sha256': sha(folder/'DONE.json')}
        rows[condition] = {str(r['id']): r for r in data}
        configs[condition] = config
    original = configs['Base_SWprefix']
    assert all(configs[c][key] == original[key] for c in configs for key in keys)
    assert configs['Base_ENprefix']['adapter'] is None and original['adapter'] is None
    assert configs['SW20_ENprefix']['protocol'] == freeze['protocol']
    assert configs['SW20_ENprefix']['adapter'] == str(args.root/freeze['adapter_relative'])
    assert configs['SW20_ENprefix']['checkpoint_step'] == 20
    assert configs['SW20_ENprefix']['adapter'] == configs['SW20_SWprefix']['adapter']
    assert sha(args.root/freeze['adapter_relative']/'adapter_model.safetensors') == freeze['adapter_sha256']
    assert configs['SW20_ENprefix']['original_evaluator_sha256'] == original['evaluator_sha256']
    assert configs['Base_ENprefix']['original_evaluator_sha256'] == original['evaluator_sha256']
    assert configs['SW20_SWprefix']['evaluator_sha256'] == original['evaluator_sha256']
    assert sha(output_root/ids['Base_SWprefix']/'ITEMS.jsonl') == freeze['native_Base_items_sha256']
    before, after = freeze['native_think_prefix'], freeze['English_think_prefix']
    for item_id, base in rows['Base_SWprefix'].items():
        for condition in rows:
            other = rows[condition][item_id]
            assert all(base[k] == other[k] for k in ['question', 'answer'])
            assert base['samples'][0]['seed'] == other['samples'][0]['seed']
            expected = base['prompt'][:-len(before)]+after if condition.endswith('ENprefix') else base['prompt']
            assert base['prompt'].endswith(before) and other['prompt'] == expected
    paired = {}
    for left, right in [('Base_SWprefix', 'Base_ENprefix'), ('Base_SWprefix', 'SW20_SWprefix'), ('Base_ENprefix', 'SW20_ENprefix'), ('SW20_SWprefix', 'SW20_ENprefix')]:
        gains, losses = [], []
        for item_id, row in rows[left].items():
            a, b = row['samples'][0]['correct'], rows[right][item_id]['samples'][0]['correct']
            if b and not a: gains.append(item_id)
            if a and not b: losses.append(item_id)
        paired[left+'__to__'+right] = {'gained_ids': gains, 'lost_ids': losses, 'delta_correct': len(gains)-len(losses)}
    job = state['job']
    hours = (datetime.fromisoformat(job['finished_at'])-datetime.fromisoformat(job['started_at'])).total_seconds()/3600
    result = {'protocol': freeze['protocol'], 'completed_at': datetime.now().isoformat(),
              'cases': cases, 'paired': paired, 'all100_only_prefix_changed': True,
              'parent_GPU_hours': hours, 'new_training': False,
              'training_delta_under_SWprefix_pp': cases['SW20_SWprefix']['single_correct']-cases['Base_SWprefix']['single_correct'],
              'training_delta_under_ENprefix_pp': cases['SW20_ENprefix']['single_correct']-cases['Base_ENprefix']['single_correct'],
              'interaction_ENminusSW_training_delta_pp': (cases['SW20_ENprefix']['single_correct']-cases['Base_ENprefix']['single_correct'])-(cases['SW20_SWprefix']['single_correct']-cases['Base_SWprefix']['single_correct']),
              'frozen_metadata_note': 'Legacy adapter:null field in frozen manifest is unused metadata copied from Base control. Runtime reads adapter_relative and checks adapter_sha256; both trained CONFIGs use same actual checkpoint20. Frozen manifest retained unchanged.',
              'scope': 'Posthoc fixed2x2 development100 diagnostic, singletrainingseed and onegeneration/question, not formal benchmark or independent confirmation. Same-prefix training deltas and interaction are descriptive, not an additive causal attribution. Official nativeSW scores retained, English-prefix results separate.',
              'summarizer_sha256': sha(Path(__file__))}
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tb = args.root/'logs/copsd_official_10lang/tensorboard/sw4b_prompt_interaction_20261022'
    writer = SummaryWriter(str(tb))
    for condition, case in cases.items():
        writer.add_scalar(condition+'/accuracy', case['single_correct']/100, 20)
    for key in ['training_delta_under_SWprefix_pp', 'training_delta_under_ENprefix_pp', 'interaction_ENminusSW_training_delta_pp']:
        writer.add_scalar(key, result[key], 20)
    writer.add_scalar('cost/this_parent_GPU_hours', hours, 20)
    writer.add_scalar('cost/this_parent_generated_tokens', cases['SW20_ENprefix']['generated_tokens'], 20)
    writer.close()
    accumulator = EventAccumulator(str(tb)).Reload()
    for condition, case in cases.items():
        assert abs(accumulator.Scalars(condition+'/accuracy')[-1].value-case['single_correct']/100) < 1e-7
    result['TensorBoard_four_accuracies_three_deltas_two_costs_read_back'] = True
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v['single_correct'] for k, v in cases.items()}))


if __name__ == '__main__':
    main()
