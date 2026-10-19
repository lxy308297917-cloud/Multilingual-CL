"""Read saved dev responses; report text diagnostics without new model inference."""
import argparse
import datetime
import json
from pathlib import Path
import re

DIGITS = r'\d'
MATH_COMMANDS = r'\\(?:frac|sqrt|sum|prod|begin|end|cdot|times|boxed|left|right|equiv|pmod|binom|log|sin|cos|tan)\b'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--language', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('runs', nargs='+', help='LABEL=existing dev result directory')
    args = parser.parse_args()
    measures = []
    for run in args.runs:
        label, directory = run.split('=', 1)
        path = Path(directory)
        cfg = json.loads((path / 'CONFIG.json').read_text())
        done = json.loads((path / 'DONE.json').read_text())
        assert cfg['language'] == args.language
        assert cfg['protocol'] == 'project_dev_single_vllm_v1'
        rows = [json.loads(line) for line in (path / 'ITEMS.jsonl').read_text().splitlines()]
        assert len(rows) == done['items'] == 100 and not done['formal_test']
        texts = [row['samples'][0]['official_generated_text'] for row in rows]
        digits = [len(re.findall(DIGITS, text)) for text in texts]
        commands = [len(re.findall(MATH_COMMANDS, text)) for text in texts]
        measures.append({'checkpoint': label, 'items': 100, 'correct': done['single_correct'],
            'responses_with_any_Unicode_decimal_digit': sum(n > 0 for n in digits),
            'total_Unicode_decimal_digits': sum(digits),
            'responses_with_selected_LaTex_math_commands': sum(n > 0 for n in commands),
            'selected_LaTex_math_command_count': sum(commands),
            'responses_with_close_think_tag': sum('</think>' in t for t in texts),
            'responses_with_literal_boxed': sum('\\boxed' in t for t in texts)})
    result = {'observed_at': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'language': args.language, 'source': 'Existing same-protocol held-out dev100 outputs; no new inference',
        'measures': measures, 'regex': {'Unicode_decimal_digits': DIGITS, 'selected_LaTex_commands': MATH_COMMANDS},
        'limitation': 'Text diagnostics, not standalone correctness or fluency scores. Numbers may be written as words; selected commands do not cover all mathematics. Counts do not identify a causal training fault.'}
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
