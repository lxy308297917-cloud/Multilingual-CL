"""CPU language analysis for all seven accepted formal cases of a 4B model.

Reuses the existing detector and keeps thinking/final sections separate.
Language ID is not a measure of fluency, proof quality or mathematical accuracy.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from build_sequential4b_matrix import accepted_evaluation
from formal4b_contract import digest, read
from sequential4b_selection import local

LABELS = {'sw': 'swh_Latn', 'te': 'tel_Telu', 'th': 'tha_Thai',
          'bn': 'ben_Beng', 'zh': 'cmn_Hani', 'ja': 'jpn_Jpan',
          'ru': 'rus_Cyrl', 'de': 'deu_Latn', 'fr': 'fra_Latn',
          'es': 'spa_Latn', 'en': 'eng_Latn'}


def analyze(root, evaluation, output_dir):
    root, evaluation, output_dir = Path(root), Path(evaluation), Path(output_dir)
    code = Path(__file__).parent
    original = read(evaluation)
    marker, plan = accepted_evaluation(root, evaluation, original['selection_record_sha256'])
    assert not output_dir.exists()
    assert output_dir.resolve().is_relative_to((root / 'logs').resolve())
    output_dir.mkdir(parents=True, exist_ok=False)
    grouped = {}
    for name, case in marker['cases'].items():
        language = plan['cases'][name]['language']
        grouped.setdefault(language, []).append((name, local(root, case['output_relative']) / 'ITEMS.jsonl'))
    assert len(grouped) == 4 and 'en' in grouped
    model = root / 'models/glotlid/model_v3.bin'
    model_sha = digest(model)
    reports = {}
    for language, inputs in grouped.items():
        label = '__label__' + LABELS[language]
        output = output_dir / (language + '.PRIVATE.json')
        command = [str(root / 'envs/copsd-official-20261015/bin/python'),
                   str(code / 'analyze_output_languages.py'), '--items',
                   *[str(path) for _, path in inputs], '--model', str(model),
                   '--target-label', label, '--output', str(output)]
        with (output_dir / (language + '.log')).open('x') as log:
            subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        report = read(output)
        assert report['model_sha256'] == model_sha and report['target_label'] == label
        assert len(report['runs']) == len(inputs)
        for name, path in inputs:
            run = report['runs'][path.parent.name]
            assert run['items_sha256'] == digest(path)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            predictions = run['predictions']
            assert len(rows) == len(predictions) == marker['cases'][name]['metrics']['items']
            assert all((row['id'], row['samples'][0]['correct']) == (prediction['id'], prediction['correct'])
                       for row, prediction in zip(rows, predictions))
            def target(prediction, part):
                value = prediction[part]
                return value.get('label') == label and value.get('prob', 0) >= .8
            reports[name] = {
                'language': language, 'target_label': label,
                'items': len(rows), 'language_summary': run['summary'],
                'formal_single_correct': marker['cases'][name]['metrics']['single_correct'],
                'correct_and_target_thinking_highconf': sum(p['correct'] and target(p, 'thinking') for p in predictions),
                'correct_and_target_final_highconf': sum(p['correct'] and target(p, 'final') for p in predictions),
                'correct_and_both_target_highconf': sum(p['correct'] and target(p, 'thinking') and target(p, 'final') for p in predictions),
                'correct_without_final_section': sum(p['correct'] and p['final']['status'] == 'no_final_section' for p in predictions),
                'analysis_relative': str(output.relative_to(root)),
                'analysis_sha256': digest(output), 'items_sha256': digest(path)}
            assert sum(p['correct'] for p in predictions) == reports[name]['formal_single_correct']
    assert set(reports) == set(marker['cases'])
    result = {'model_id': marker['model_id'], 'evaluation_marker_relative': str(evaluation.relative_to(root)),
              'evaluation_marker_sha256': digest(evaluation),
              'selection_record_sha256': marker['selection_record_sha256'],
              'language_model_sha256': model_sha,
              'detector_source_sha256': digest(code / 'analyze_output_languages.py'),
              'collector_source_sha256': digest(Path(__file__)),
              'cases': reports, 'GPU_started': False,
              'scope': 'All seven saved formal outputs analyzed on CPU with the same GlotLID model and detector. Original mathematical scores retained, including correct answers without final sections; short numeric finals may be unclassifiable. Language ID is not calibrated fluency, reasoning quality or evidence of causal transfer.'}
    with (output_dir / 'COMPLETED.PRIVATE.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.evaluation, args.output_dir)
    print(json.dumps({'model_id': result['model_id'], 'language_cases': len(result['cases']), 'GPU_started': False}))


if __name__ == '__main__':
    main()
