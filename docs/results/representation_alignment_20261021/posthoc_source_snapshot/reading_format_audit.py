"""Post-hoc format audit on frozen cached outputs; never changes primary gates."""
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'outputs/Multilingual-CL-paired-deadline/experiments/paired_capability_sparse_v1'
sys.path.insert(0, str(CODE))
from compare_dev import paired_ci

HOME = Path('/data0/users/lixiaoyang/models/checkpoint')


def extract(text, source):
    s = text.strip()
    # Do not interpret the article 'A' in ordinary prose as an option letter.
    if len(s) == 1 and s.upper() in 'ABCD':
        return s.upper(), 'bare_letter'
    if len(set(re.findall(r'^\s*([A-D])[.)]\s', s, flags=re.M))) > 1:
        return None, 'ambiguous_multiple_option_lines'
    match = re.match(r'^\s*(?:\(([A-D])\)|([A-D])[.)\]:：])(?:\s|$)', s, re.I)
    if match:
        return (match[1] or match[2]).upper(), 'leading_option_label'
    match = re.match(r'^\s*(?:the\s+)?(?:(?:correct|best)\s+)?(?:answer|option)\s*(?:is\s*)?[:：]?\s*\(?([A-D])\)?(?:[.)\s]|$)', s, re.I)
    if match:
        return match[1].upper(), 'explicit_answer_prefix'
    section = source['prompt'].rsplit('Question:', 1)[-1]
    options = re.findall(r'^([A-D])\.\s*(.*)$', section, flags=re.M)
    assert [k for k, _ in options] == list('ABCD')
    normalize = lambda value: re.sub(r'\s+', ' ', value).strip().rstrip('.').casefold()
    matches = [k for k, option in options if normalize(s) == normalize(option)]
    if len(matches) == 1:
        return matches[0], 'exact_unique_option_content'
    return None, 'unparsed'


def main():
    protocol = json.loads((CODE / 'comparison_protocol.v1.json').read_text())
    manifest = json.loads(Path(protocol['development']['manifest']).read_text())
    source_path = Path(manifest['partitions']['en_reading']['path'])
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == manifest['partitions']['en_reading']['sha256']
    sources = {row['id']: row for row in map(json.loads, source_path.read_text().splitlines())}
    folders = {'sparse-' + m: HOME / 'paired_capability_sparse_20261021/comparison/matched_v1' / ('dev_' + m) for m in ['Base', 'A2048', 'D2048', 'R2048']}
    folders.update({'align-' + m: HOME / 'representation_alignment_20261021/development' / m for m in ['Base', 'A', 'B', 'C', 'D']})
    summaries, repaired, examples = {}, {}, []
    for name, folder in folders.items():
        done = json.loads((folder / 'DONE.json').read_text())
        raw = folder / 'PREDICTIONS.jsonl'
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == done['predictions_sha256']
        rows = [r for r in map(json.loads, raw.read_text().splitlines()) if r['task'] == 'en_reading']
        assert len(rows) == 256
        parsed_rows, methods = [], {}
        format_correct, strict_correct, content_correct, unparsed = 0, 0, 0, 0
        for row in rows:
            source = sources[row['id']]
            letter, method = extract(row['text'], source)
            methods[method] = methods.get(method, 0) + 1
            correct = letter == source['reference']
            strict = row['scores']['strict_letter']
            content_correct += int(correct)
            strict_correct += strict
            unparsed += int(letter is None)
            format_correct += int(correct and not strict)
            replacement = copy.deepcopy(row)
            replacement['scores'] = {'strict_letter': float(correct)}
            parsed_rows.append(replacement)
            if correct and not strict and sum(x['model'] == name for x in examples) < 4:
                examples.append({'model': name, 'id': row['id'], 'gold': source['reference'], 'output': row['text'], 'primary_strict_score': strict, 'posthoc_choice': letter, 'method': method, 'capped': row['capped']})
        repaired[name] = parsed_rows
        summaries[name] = {'n': 256, 'frozen_primary_strict_letter_percent': strict_correct / 256 * 100, 'posthoc_unambiguous_choice_percent': content_correct / 256 * 100, 'content_correct_but_strict_format_failed': format_correct, 'unparsed': unparsed, 'parser_methods': methods}
    pairs = [('sparse-D2048', 'sparse-A2048'), ('sparse-D2048', 'sparse-R2048'), ('align-C', 'align-B'), ('align-D', 'align-B'), ('align-C', 'align-A'), ('align-D', 'align-A')]
    intervals = {}
    for left, right in pairs:
        result = paired_ci(repaired[left], repaired[right])
        result['metric'] = 'POSTHOC_unambiguous_option_choice'
        intervals[left + '-' + right] = result
    result = {'status': 'posthoc diagnostic; not a replacement primary metric, confirmation set, or model-selection gate', 'why': "Frozen strict metric counts correct 'A. option text' as incorrect; original explicit fallback also misses leading punctuated labels. Scores near zero cannot be interpreted as reading ability near zero.", 'original_predictions_and_primary_decisions_preserved': True, 'new_gpu_forward_calls': 0, 'original_upgrade_gates_unchanged': True, 'parser_limits': 'Conservative anchored option labels, explicit prefixes, or exactly one full option text; unresolved or multiple option-line answers remain unparsed. This audit does not remove the generation cap or establish general reading competence.', 'source_sha256': manifest['partitions']['en_reading']['sha256'], 'models': summaries, 'paired_source_cluster_intervals': intervals, 'examples_in_original_source_order': examples}
    (ROOT / 'outputs/READING_FORMAT_POSTHOC_AUDIT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
