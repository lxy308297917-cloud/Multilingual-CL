"""Create/verify a private identity lock; never publish its actual fingerprints."""
import argparse
import hashlib
import json
from pathlib import Path

COMMIT = '4f4410730e9b6775a30d20f4b6e141cba2e87590'
FILES = ['evaluate_vllm.py', 'evaluate_transformers.py', 'freeze_vllm_eval.py',
         'run_vllm_math_eval.sh', 'run_sw_afrimgsm_vllm_pair.sh',
         'compat/vllm011_uuid/sitecustomize.py']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['create', 'verify'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--input', type=Path)
    p.add_argument('--source', type=Path)
    args = p.parse_args()
    root, code = args.root.resolve(), Path(__file__).resolve().parent
    official = root / 'COPSD-official-20261015'
    upstream_path = root / 'logs/copsd_official_10lang/eval_upstream_4f441073.SOURCE_LOCK.json'
    upstream = json.loads(upstream_path.read_text())
    if upstream['commit'] != COMMIT:
        raise ValueError('Wrong pinned upstream source lock')
    sources = ['african_langs_eval/evaluate_math.py', 'polymath_eval/evaluate_math.py',
               'african_langs_eval/run_afrimgsm_one_lang_all_ckpts_1.7b.sh']
    for name in sources:
        if digest(official/name) != upstream['files'][name]['sha256']:
            raise ValueError('Local export differs from pinned upstream file: '+name)
    if args.mode == 'create':
        if args.lock.exists():
            raise FileExistsError(args.lock)
        inputs = [root / f'data/math_single_10lang_20261015/mgsm_rev2/MGSM-Rev2/mgsm_{lang}.tsv'
                  for lang in 'sw te th bn zh ja ru de fr es en'.split()]
        inputs += [root / f'data/copsd_official_20261015/polymath_project_eval_v1/polymath_all500_{lang}.json'
                   for lang in 'sw te th bn zh ja ru de fr es'.split()]
        inputs += [root / 'data/copsd_official_20261015/afrimgsm_8e4268d2b949/data/swa/test.tsv',
                   root / 'data/copsd_official_20261015/gsm8k_project_en_zeroshot_v1/gsm8k_en_test_1319.json']
        parent = json.loads((root / 'logs/copsd_official_10lang/th70_resume_uuid_20261019_v2.MANIFEST.json').read_text())
        model_lock = parent['model_lock']
        if digest(Path(model_lock['path'])) != model_lock['sha256']:
            raise ValueError('Model identity lock changed')
        lock = {'official_commit': COMMIT, 'upstream_identity_lock_sha256': digest(upstream_path),
                'code_sha256': {n: digest(code/n) for n in FILES},
                'source_sha256': {n: digest(official/n) for n in sources},
                'input_sha256': {str(n.relative_to(root)): digest(n) for n in inputs},
                'model_lock': model_lock}
        args.lock.parent.mkdir(parents=True, exist_ok=True)
        with args.lock.open('x') as stream:
            json.dump(lock, stream, indent=2)
            stream.write('\n')
        print('Private runtime/source/model lock created for23 frozen benchmark files.')
    else:
        lock = json.loads(args.lock.read_text())
        if lock['official_commit'] != COMMIT:
            raise ValueError('Wrong official lock')
        if digest(upstream_path) != lock['upstream_identity_lock_sha256']:
            raise ValueError('Changed pinned upstream identity lock')
        for name, expected in lock['code_sha256'].items():
            if digest(code/name) != expected:
                raise ValueError('Changed evaluator code: '+name)
        for name, expected in lock['source_sha256'].items():
            if digest(official/name) != expected:
                raise ValueError('Changed official evaluation source: '+name)
        if digest(Path(lock['model_lock']['path'])) != lock['model_lock']['sha256']:
            raise ValueError('Changed model lock')
        if args.input is None or args.source is None:
            raise ValueError('Input/source required for verification')
        relative = str(args.input.resolve().relative_to(root))
        if digest(args.input) != lock['input_sha256'][relative]:
            raise ValueError('Changed benchmark input')
        source_relative = str(args.source.resolve().relative_to(official))
        if digest(args.source) != lock['source_sha256'][source_relative]:
            raise ValueError('Wrong official scorer source')
        print('Frozen evaluator, original official functions, model and benchmark input verified.')


if __name__ == '__main__':
    main()
