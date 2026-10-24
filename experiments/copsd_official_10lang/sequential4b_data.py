"""Bind a frozen 4B stage to its language's existing disjoint question pool.

This module prepares data only. It never chooses languages or starts training.
Reverse orders retain the same language-to-pool mapping as matched singles.
"""
import argparse
import json
from pathlib import Path

from prepare_sequential_pools import ENGLISH_FIELDS, read, sha


def pool_rows(root, manifest, expected_sha, language, index):
    """Read immutable source rows, retaining the manifest's exact ID order."""
    root, manifest = Path(root), Path(manifest)
    assert sha(manifest) == expected_sha
    pools = read(manifest)
    assert type(index) is int and 0 <= index < 3
    sets = [set(ids) for ids in pools['pool_ids']]
    assert all(len(ids) == len(unique) == 900
               for ids, unique in zip(pools['pool_ids'], sets))
    assert sum(map(len, sets)) == len(set.union(*sets)) == 2700
    assert not set.union(*sets) & set(pools['development_ids'])
    source = Path(pools['inputs'][language]['path'])
    assert source.resolve().is_relative_to(root.resolve())
    assert sha(source) == pools['inputs'][language]['sha256']
    rows = read(source)
    lookup = {row['problem_id']: row for row in rows}
    assert len(lookup) == len(rows)
    result = [lookup[identifier] for identifier in pools['pool_ids'][index]]
    # Preserve the audited repair for JA, without silently rewriting any row.
    if language == 'ja':
        code = Path(__file__).parent
        repaired = root / 'data/copsd_official_20261015/adapted_v3_ja_prefix_repair/project_train_2900_ja.json'
        assert sha(repaired) == read(code / 'ja_repair_runtime_20261020.CONFIG.json')['data_sha256']
        fixed = {row['problem_id']: row for row in read(repaired)}
        result = [fixed[identifier] for identifier in pools['pool_ids'][index]]
    assert all(isinstance(row['problem_' + language], str)
               and row['problem_' + language].strip() for row in result)
    reference = Path(pools['inputs']['sw']['path'])
    assert sha(reference) == pools['inputs']['sw']['sha256']
    english = {row['problem_id']: row for row in read(reference)}
    assert all(all(row[key] == english[row['problem_id']][key]
                   for key in ENGLISH_FIELDS) for row in result)
    assert len(result) >= 20 * 32
    return result


def bind(root, selection_path, selection_sha, direction, stage, output,
         language=None, role='sequential_stage'):
    from sequential4b_selection import local, selection, stage_context

    root, output = Path(root), Path(output)
    selected = selection(root, Path(selection_path), selection_sha)
    context = stage_context(selected, direction, stage, language, role)
    rows = pool_rows(root, local(root, selected['pool_manifest_relative']),
                     selected['pool_manifest_sha256'], context['language'],
                     context['pool_index'])
    assert output.resolve().is_relative_to((root / 'data').resolve())
    manifest = output.with_name(output.name + '.BINDING.PRIVATE.json')
    # Exclusive writes preserve previous materializations and their provenance.
    assert not output.exists() and not manifest.exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(rows, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    receipt = {'context': context, 'training_data_relative': str(output.relative_to(root)),
               'training_data_sha256': sha(output), 'rows': len(rows),
               'selection_record_relative': str(Path(selection_path).relative_to(root)),
               'selection_record_sha256': selection_sha,
               'pool_manifest_relative': selected['pool_manifest_relative'],
               'question_ids': [row['problem_id'] for row in rows],
               'development_overlap': 0, 'english_fields_unchanged': True}
    with manifest.open('x') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--selection-sha256', required=True)
    parser.add_argument('--direction', choices=['matched', 'forward', 'reverse'], required=True)
    parser.add_argument('--stage', type=int, required=True)
    parser.add_argument('--language')
    parser.add_argument('--role', choices=['matched_single_reference', 'sequential_stage'],
                        default='sequential_stage')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = bind(args.root, args.selection, args.selection_sha256,
                  args.direction, args.stage, args.output, args.language, args.role)
    print(json.dumps({'data_prepared': True, 'GPU_started': False,
                      'rows': result['rows'], 'context': result['context']}))


if __name__ == '__main__':
    main()
