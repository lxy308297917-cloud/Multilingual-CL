"""Build private dev prompts with the pinned official collator, without inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--language', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    lang = args.language.lower()
    assert lang in 'sw te th bn zh ja ru de fr es'.split()
    assert not args.output.exists()
    source = args.root / 'COPSD-official-20261015'
    sys.path.insert(0, str(source))
    from multilingual_data_collator import MultilingualSelfDistillationDataCollator
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.root / 'models/Qwen3-1.7B', local_files_only=True)
    collator = MultilingualSelfDistillationDataCollator(tokenizer=tokenizer, max_length=20000,
        reason_first=False, student_enable_thinking=True, include_problem_en=True,
        include_reference_solution_en=True)
    data_path = args.root / f'data/copsd_official_20261015/adapted_v2_ordered/project_dev_100_{lang}.json'
    data = json.loads(data_path.read_text())
    assert len(data) == 100
    records = []
    for row in data:
        assert row['solution'].strip() and row['problem'].strip() and row[f'problem_{lang}'].strip()
        encoded = collator([{**row, 'target_lang': lang.upper()}])
        student = encoded['student_prompts'][0].tolist()
        teacher = encoded['teacher_prompts'][0].tolist()
        assert len(student) < 20000 and len(teacher) < 20000
        assert all(v == 1 for v in encoded['teacher_prompt_attention_mask'][0].tolist())
        assert len(teacher) > len(student)
        records.append({'id': row['problem_id'], 'answer': row['Answer'],
            'student_prompt_token_ids': student, 'teacher_prompt_token_ids': teacher})
    args.output.mkdir(parents=True)
    (args.output / 'PROMPTS.PRIVATE.json').write_text(json.dumps(records, ensure_ascii=False) + '\n')
    config = {'language': lang, 'items': 100, 'formal_test': False,
        'protocol': 'privileged_teacher_conditioning_diagnostic_v1',
        'model': str(args.root / 'models/Qwen3-1.7B'), 'source': str(source),
        'collator_sha256': hashlib.sha256((source / 'multilingual_data_collator.py').read_bytes()).hexdigest(),
        'language_config_sha256': hashlib.sha256((source / 'language_config.py').read_bytes()).hexdigest(),
        'input_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
        'prompts_sha256': hashlib.sha256((args.output / 'PROMPTS.PRIVATE.json').read_bytes()).hexdigest(),
        'max_length': 20000, 'reason_first': False, 'student_enable_thinking': True,
        'include_problem_en': True, 'include_reference_solution_en': True,
        'scope': 'Official collator conditioning diagnostic. Teacher receives reference solution; its conditional score is not a fair unprivileged benchmark or a measurement of every on-policy teacher token.'}
    (args.output / 'CONFIG.PRIVATE.json').write_text(json.dumps(config, indent=2) + '\n')
    summary = {'language': lang, 'items': 100, 'inference_performed': False,
        'student_prefix_tokens': sum(len(r['student_prompt_token_ids']) for r in records),
        'teacher_prefix_tokens': sum(len(r['teacher_prompt_token_ids']) for r in records),
        'all_prefixes_untruncated': True, 'official_collator_used_directly': True,
        'teacher_receives_English_problem_and_reference': True,
        'student_receives_target_problem_only': True, 'formal_test': False,
        'GPU_validation': 'Not yet run; private prompts prepared only.'}
    (args.output / 'PREPARED.PUBLIC.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
