"""Post-hoc local GSM8K flexible-extract diagnostic; no primary gate changes."""
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'outputs/Multilingual-CL-paired-deadline/experiments/paired_capability_sparse_v1'
sys.path.insert(0, str(CODE))
from compare_dev import paired_ci

H = Path('/data0/users/lixiaoyang')
TASK = H / 'reproductions/sw_d2_20260921/harness_local/lm_eval/tasks/gsm8k/gsm8k.yaml'
FILTER = TASK.parents[2] / 'filters/extraction.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    task = yaml.safe_load(TASK.read_text())
    config = next(f for f in task['filter_list'] if f['name'] == 'flexible-extract')['filter'][0]
    assert config['group_select'] == -1
    pattern = re.compile(config['regex_pattern'])
    ignored = task['metric_list'][0]['regexes_to_ignore']
    def normalize(value):
        for expression in ignored:
            value = re.sub(expression, '', value)
        return value.lower()
    protocol = json.loads((CODE / 'comparison_protocol.v1.json').read_text())
    manifest = json.loads(Path(protocol['development']['manifest']).read_text())
    info = manifest['partitions']['en_math']
    assert sha(info['path']) == info['sha256']
    sources = {r['id']: r for r in map(json.loads, Path(info['path']).read_text().splitlines())}
    folders = {'sparse-' + m: H / 'models/checkpoint/paired_capability_sparse_20261021/comparison/matched_v1' / ('dev_' + m) for m in ['Base', 'A2048', 'D2048', 'R2048']}
    folders.update({'align-' + m: H / 'models/checkpoint/representation_alignment_20261021/development' / m for m in ['Base', 'A', 'B', 'C', 'D']})
    results, repaired, details = {}, {}, []
    for name, folder in folders.items():
        done = json.loads((folder / 'DONE.json').read_text())
        raw = folder / 'PREDICTIONS.jsonl'
        assert sha(raw) == done['predictions_sha256']
        rows = [r for r in map(json.loads, raw.read_text().splitlines()) if r['task'] == 'en_math']
        assert len(rows) == len(sources) == 128
        correct = primary = no_extract = improved = lost = 0
        repaired[name] = []
        for row in rows:
            matches = pattern.findall(row['text'])
            if matches:
                value = matches[-1]
                if isinstance(value, tuple):
                    value = next((v for v in value if v), '[invalid]')
                value = value.strip()
            else:
                value = '[invalid]'
                no_extract += 1
            score = float(normalize(value) == normalize(sources[row['id']]['reference']))
            old_score = row['scores']['numeric_equivalence']
            correct += score
            primary += old_score
            improved += int(score and not old_score)
            lost += int(old_score and not score)
            item = copy.deepcopy(row)
            item['scores'] = {'numeric_equivalence': score}
            repaired[name].append(item)
            details.append({'model': name, 'id': row['id'], 'gold': sources[row['id']]['reference'], 'flexible_extract': value, 'flexible_exact_match': score, 'primary_numeric_equivalence': old_score, 'capped': row['capped']})
        results[name] = {'n': 128, 'primary_format_sensitive_numeric_percent': primary / 128 * 100, 'POSTHOC_local_GSM8K_flexible_extract_percent': correct / 128 * 100, 'flex_correct_primary_incorrect': improved, 'primary_correct_flex_incorrect': lost, 'no_numeric_match': no_extract, 'capped': sum(r['capped'] for r in rows)}
    pairs = [('sparse-D2048', 'sparse-A2048'), ('sparse-D2048', 'sparse-R2048'), ('align-C', 'align-B'), ('align-D', 'align-B'), ('align-C', 'align-A'), ('align-D', 'align-A')]
    intervals = {}
    for left, right in pairs:
        result = paired_ci(repaired[left], repaired[right])
        result['metric'] = 'POSTHOC_local_GSM8K_flexible_extract'
        intervals[left + '-' + right] = result
    report = {'status': 'POSTHOC format sensitivity diagnostic, not a preregistered primary, confirmation set, or selection criterion', 'why': 'Cached dense models give correct plain-language numbers but omit boxed/explicit-colon format, causing frozen primary scores of zero. Apply the pre-existing local harness flexible-extract regex and metric cleanup uniformly to all nine endpoints.', 'local_task_file': str(TASK), 'local_task_sha256': sha(TASK), 'regex_filter_source': str(FILTER), 'regex_filter_source_sha256': sha(FILTER), 'regex': config['regex_pattern'], 'cleanup_regexes': ignored, 'script_sha256': sha(__file__), 'limits': 'Last-number extraction can select an intermediate number, mishandle scientific notation/fractions, or disagree with a valid earlier boxed answer. These are held-out TRAIN-side chat outputs, not the harness official test/few-shot scores. No change to prompts, caps, original predictions, primary decisions, or upgrade release.', 'new_gpu_forward_calls': 0, 'models': results, 'paired_source_cluster_intervals': intervals}
    (ROOT / 'outputs/MATH_FORMAT_POSTHOC_AUDIT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    with (ROOT / 'outputs/MATH_FORMAT_POSTHOC_ITEMS.jsonl').open('w') as stream:
        for row in details:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
