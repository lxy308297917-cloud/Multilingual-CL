"""Bind ten completed primary models to their frozen source dependencies.

Read-only CPU audit; no model loading, training or data modification.
The TE predecessor is included because checkpoint5 was retained.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


PLANS = {
    'sw': 'sw4b_server_pilot20_frozen_20261022.PUBLIC.json',
    'te': 'te4b_resume5_frozen_20261024.PUBLIC.json',
    'th': 'th4b_single20_frozen_20261024.PUBLIC.json',
    'bn': 'bn4b_server20_frozen_20261023.PUBLIC.json',
    'zh': 'zh4b_server20_frozen_20261023.PUBLIC.json',
    'ja': 'ja4b_server20_alloc_v2_frozen_20261023.PUBLIC.json',
    'ru': 'ru4b_single20_frozen_20261024.PUBLIC.json',
    'de': 'de4b_single20_frozen_20261024.PUBLIC.json',
    'fr': 'fr4b_single20_frozen_20261024.PUBLIC.json',
    'es': 'es4b_single20_frozen_20261024.PUBLIC.json',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def audit(root):
    code = Path(__file__).resolve().parent
    project, installed, plans = {}, {}, {}

    def check_plan(name):
        path = code / name
        value = read(path)
        assert value['official_commit'] == '4f4410730e9b6775a30d20f4b6e141cba2e87590'
        assert value['model_revision'] == '1cfa9a7208912126459214e8b04321603b3df60c'
        for relative, expected in value['code_sha256'].items():
            assert Path(relative).name == relative
            assert digest(code / relative) == expected, relative
            assert relative not in project or project[relative] == expected
            project[relative] = expected
        for relative, expected in value['installed_source_sha256'].items():
            assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
            assert digest(root / relative) == expected, relative
            assert relative not in installed or installed[relative] == expected
            installed[relative] = expected
        plans[name] = digest(path)
        return value

    models = {}
    for language, name in PLANS.items():
        value = check_plan(name)
        formal = read(code / f'primary4b_{language.upper()}20_formal_frozen_20261024_v1.LOCAL.json')
        adapters = {item['adapter_relative'] for item in formal['cases'].values()}
        assert len(adapters) == 1
        checkpoint = root / adapters.pop()
        assert checkpoint.parent == root / value['checkpoint_relative']
        state = read(checkpoint / 'trainer_state.json')
        done_path = checkpoint.parent / 'SERVER_PILOT20_UPDATES_DONE.json'
        done = read(done_path)
        parent = read(checkpoint.parent / 'SERVER_PARENT_DONE.json')
        assert state['global_step'] == done['global_step'] == done['completed_updates'] == 20
        assert done['effective_batch'] == 32 and done['scheduler_horizon'] == 100
        assert done['base_unchanged_bitwise'] and done['teacher_via_disabled_original_Base']
        assert parent['training_exit_code'] == 0 and parent['two_role_job_completed']
        args = value['train_arguments']
        assert args[args.index('--train_language') + 1].lower() == language
        models[language] = {'training_plan': name, 'training_plan_sha256': plans[name],
            'adapter_sha256': next(iter(formal['cases'].values()))['adapter_sha256'],
            'completed_updates': 20, 'completion_receipt_sha256': digest(done_path),
            'trainer_state_sha256': digest(checkpoint / 'trainer_state.json'),
            'train_arguments': args, 'teacher_disabled_adapter_and_frozen_Base_recorded': True}
        if language == 'te':
            predecessor = check_plan('te4b_server20_frozen_20261024.PUBLIC.json')
            assert root / value['resume_checkpoint_relative'] == root / predecessor['checkpoint_relative'] / 'checkpoint-5'
            models[language]['retained_predecessor_plan'] = 'te4b_server20_frozen_20261024.PUBLIC.json'
    return {'observed_at': datetime.datetime.now().astimezone().isoformat(),
        'official_commit': '4f4410730e9b6775a30d20f4b6e141cba2e87590',
        'model_revision': '1cfa9a7208912126459214e8b04321603b3df60c',
        'ten_primary_models': models, 'eleven_training_plans_sha256': plans,
        'project_dependencies_sha256': project,
        'installed_dependencies_sha256': {str(Path(key).relative_to('envs/copsd-vllm011-20261018/lib/python3.10/site-packages')): value for key, value in installed.items()},
        'all_frozen_source_dependencies_match': True,
        'collector_sha256': digest(Path(__file__)), 'GPU_work_started_or_changed': False,
        'scope': 'Completed original2700 primary models only, plus retained TE predecessor. Confirms current files match frozen hashes and saved completion receipts; does not retroactively prove every historical runtime event, reference semantic correctness, official100-step equivalence or unfinished formal evaluation. Official source export and complete runtime patch proof remain in REPRODUCTION_SOURCE_AUDIT_20261026.PUBLIC.json.',
        'full_goal_complete': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    report = audit(args.root.resolve())
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'models': len(report['ten_primary_models']),
        'plans': len(report['eleven_training_plans_sha256']),
        'project_dependencies': len(report['project_dependencies_sha256']),
        'installed_dependencies': len(report['installed_dependencies_sha256'])}))


if __name__ == '__main__':
    main()
