"""CPU regression for the observed systemd-start/state-file publication race."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    source = Path(__file__).with_name('continue_sequential4b_direction_v2.py')
    tree = ast.parse(source.read_text())
    parent = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    node = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == 'wait_queue')
    passed = []
    for initial in ['missing', 'partial_json']:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / 'gpu-queue/queues/fixture/state.json'
            state.parent.mkdir(parents=True)
            if initial == 'partial_json':
                state.write_text('{')
            observations = []

            def query(command, text):
                assert command == ['systemctl', '--user', 'show',
                                   'gpu-queue@fixture.service', '--property=MainPID,ActiveState']
                observations.append('verified_monitor_query')
                return 'MainPID=' + str(os.getpid()) + '\nActiveState=active\n'

            def sleep(seconds):
                assert seconds == 1 and observations == ['verified_monitor_query']
                state.write_text(json.dumps({'phase': 'completed', 'job': {'exit_code': 0}}))

            namespace = dict(root=root, json=json, Path=Path,
                             subprocess=SimpleNamespace(check_output=query),
                             time=SimpleNamespace(sleep=sleep))
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
            namespace['wait_queue']('fixture')
            assert observations == ['verified_monitor_query']
            passed.append(initial)
    result = dict(CPU_regression_passed=True, cases=passed,
                  continuation_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  live_monitor_required=True, new_queue_submissions=0, GPU_started=False,
                  scope='Executes the actual v2 wait function with a missing/partial state file and a verified-live monitor fixture. No GPU task or system service is started or changed.')
    with a.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
