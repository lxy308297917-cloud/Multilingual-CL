"""Export audited completed followup cases to separate TensorBoard runs, once."""
import argparse
import json
from pathlib import Path
from collect_followup200 import audit_case, read, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--case', required=True)
    a = p.parse_args()
    code = Path(__file__).parent
    freeze = code / 'followup200_frozen_20261021.PUBLIC.json'
    frozen = read(freeze)
    for name, expected in frozen['runtime_code_sha256'].items(): assert sha(code / name) == expected, name
    evaluations, cost = audit_case(a.root, code, frozen, sha(freeze), a.case)
    output = a.root / 'logs/copsd_official_10lang/tensorboard/followup200_20261021_v1' / a.case
    output.mkdir(parents=True, exist_ok=True)
    pending = []
    for lang, item in evaluations.items():
        marker = output / f'EXPORTED_{lang}.json'
        identity = {'protocol': frozen['protocol'], 'source': item['source'],
                    'data_freeze_sha256': sha(freeze), 'checkpoint_step': 0 if a.case == 'base' else 20}
        if marker.exists():
            assert read(marker) == identity
            assert list(output.glob('events.out.tfevents.*'))
        else: pending.append((lang, item, marker, identity))
    if pending:
        from torch.utils.tensorboard import SummaryWriter
        step = 0 if a.case == 'base' else 20
        with SummaryWriter(str(output)) as writer:
            for lang, item, marker, identity in pending:
                m = item['metrics']; prefix = 'eval/' + lang + '/'
                for tag, value in {
                    'accuracy': m['single_accuracy'], 'normal_termination_rate': m['normal_terminations'] / 200,
                    'truncation_rate': m['truncated'] / 200,
                    'answer_extraction_failure_rate': m['answer_extraction_failures'] / 200,
                    'generated_tokens': m['generated_tokens'],
                    'generation_tokens_per_second': m['generated_tokens'] / m['generation_seconds'],
                }.items(): writer.add_scalar(prefix + tag, value, step)
                writer.add_text('protocol/' + lang, frozen['protocol'] + '; n=1; items200; limit8192; native64; ' + identity['source']['run_id'], step)
                writer.flush()
                with marker.open('x') as stream:
                    json.dump(identity, stream, indent=2); stream.write('\n')
            writer.add_scalar('cost/unique_parent_GPU_hours', cost['parent_GPU_hours_counted_once'], step)
    print(json.dumps({'case': a.case, 'exported_languages': [x[0] for x in pending],
                      'existing_languages_skipped': len(evaluations) - len(pending),
                      'TensorBoard_relative': str(output.relative_to(a.root)), 'new_inference_or_training': False}))


if __name__ == '__main__': main()
