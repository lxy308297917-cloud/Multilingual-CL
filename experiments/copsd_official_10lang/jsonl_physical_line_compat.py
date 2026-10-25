"""Run CPU result analysis with physical-LF JSONL records.

A generated JSON string may legally contain U+2028/U+0085. str.splitlines()
incorrectly treats those characters as record delimiters. Preserve canonical
frozen scripts and raw bytes; only adapt splitlines on ITEMS.jsonl text under
this project's log directory. No model generation or scoring is changed.

Usage: python jsonl_physical_line_compat.py SCRIPT [original SCRIPT arguments]
The SCRIPT must receive --root so the data boundary is explicit.
"""
from pathlib import Path
import runpy
import sys
import subprocess


class PhysicalJSONLText(str):
    def splitlines(self, keepends=False):
        if not self:
            return []
        parts = self.split('\n')
        trailing = parts[-1] == ''
        if trailing:
            parts.pop()
        if keepends:
            return [part + ('\n' if i < len(parts) - 1 or trailing else '')
                    for i, part in enumerate(parts)]
        return parts


def install(root):
    boundary = (Path(root) / 'logs/copsd_official_10lang').resolve()
    original_read_text = Path.read_text

    def read_text(path, *args, **kwargs):
        value = original_read_text(path, *args, **kwargs)
        if path.name == 'ITEMS.jsonl' and boundary in path.resolve().parents:
            return PhysicalJSONLText(value)
        return value

    Path.read_text = read_text
    return original_read_text


def wrap_analysis_children(root):
    original_run = subprocess.run
    shim = Path(__file__).resolve()
    allowed = {
        'collect_sequential4b_evaluation.py',
        'record_sequential4b_evaluation_tensorboard.py',
        'analyze_sequential4b_languages.py',
        'build_sequential4b_matrix.py',
        'compare_sequential4b_matched_single.py',
    }

    def run(command, *args, **kwargs):
        if isinstance(command, (list, tuple)) and len(command) > 1:
            target = Path(command[1])
            if target.name in allowed and target.resolve().parent == shim.parent:
                command = [command[0], str(shim), '--project-root', str(root),
                           *command[1:]]
        return original_run(command, *args, **kwargs)

    subprocess.run = run


def main():
    values = sys.argv[1:]
    if values[0] == '--project-root':
        root, target, *arguments = values[1:]
    else:
        target, *arguments = values
        root = arguments[arguments.index('--root') + 1]
    install(root)
    wrap_analysis_children(root)
    sys.argv = [target, *arguments]
    runpy.run_path(target, run_name='__main__')


if __name__ == '__main__':
    main()
