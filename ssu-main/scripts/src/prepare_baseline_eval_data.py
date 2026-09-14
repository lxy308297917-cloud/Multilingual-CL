#!/usr/bin/env python3
"""Reconstruct the frozen Igbo evaluation inputs from immutable public revisions.
CPU only. No training, evaluation, model downloads, or changes outside output/cache.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import shutil
import time
import urllib.error


def digest(rows, fields):
    h = hashlib.sha256()
    for row in rows:
        obj = {k: row[k] for k in fields}
        h.update((json.dumps(obj, ensure_ascii=False, sort_keys=True,
                             separators=(',', ':')) + '\n').encode())
    return h.hexdigest()


def read_rows(path):
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    if path.suffix == '.arrow':
        from datasets import Dataset
        return Dataset.from_file(str(path)).to_list()
    return [json.loads(line) for line in path.read_text().split('\n') if line.strip()]


def download(source, cache):
    # Immutable revision is part of both the URL and the cache key.
    if len(source['revision']) != 40 or any(c not in '0123456789abcdef' for c in source['revision']):
        raise ValueError('A full immutable revision SHA is required')
    path = cache / source['repo'] / source['revision'] / source['file']
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{source['file']}"
        tmp = path.with_name(path.name + '.part')
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=60) as r, tmp.open('wb') as w:
                    shutil.copyfileobj(r, w)
                tmp.replace(path)
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
            finally:
                tmp.unlink(missing_ok=True)
    if source.get('sha256') and hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
        raise ValueError(f'Public source checksum mismatch: {path}')
    return path


def select_rows(item, rows):
    rule = item['selection']
    if rule == 'frozen_ids':
        by_id = {}
        for row in rows:
            if row['id'] in by_id:
                raise ValueError(f"Duplicate source ID: {row['id']}")
            by_id[row['id']] = row
        rows = [by_id[sample_id] for sample_id in item['ids']]
    elif rule == 'pandas_sample_500_seed42':
        import pandas as pd
        rows = pd.DataFrame(rows).sample(n=500, random_state=42).to_dict('records')
    elif rule != 'all':
        raise ValueError(rule)
    if item.get('rename'):
        rows = [{new: row[old] for new, old in item['rename'].items()} for row in rows]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-root', required=True, type=Path)
    ap.add_argument('--cache-dir', type=Path)
    ap.add_argument('--spec', type=Path, default=Path(__file__).resolve().parents[2] / 'configs/ig_baseline_data_v1.json')
    args = ap.parse_args()
    spec = json.loads(args.spec.read_text())
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache = (args.cache_dir or root / 'public_downloads').resolve()
    # A changed/interrupted rebuild must not leave an old success receipt.
    receipt = root / 'DATA_VERIFIED.json'
    receipt.unlink(missing_ok=True)
    manifest = {'schema_version': 1, 'evaluation_inputs_inherited': {}, 'datasets': []}
    for item in spec['datasets']:
        rows = []
        for source in item['sources']:
            rows.extend(read_rows(download(source, cache)))
        rows = select_rows(item, rows)
        actual = digest(rows, item['fields'])
        if len(rows) != item['count'] or actual != item['content_sha256']:
            raise ValueError(f"{item['path']}: identity mismatch ({len(rows)} rows, {actual}); refusing new sample set")
        target = root / item['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        # Serialize only consumed/audited fields, retaining their types and order.
        rows = [{k: row[k] for k in item['fields']} for row in rows]
        tmp = target.with_name(target.stem + '.building' + target.suffix)
        if target.suffix == '.parquet':
            import pyarrow as pa
            import pyarrow.parquet as pq
            pq.write_table(pa.Table.from_pylist(rows), tmp)
        elif target.suffix == '.arrow':
            import pyarrow as pa
            table = pa.Table.from_pylist(rows)
            with pa.OSFile(str(tmp), 'wb') as sink, pa.ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)
        else:
            tmp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        if digest(read_rows(tmp), item['fields']) != actual:
            raise ValueError(f'Serialization changed records: {target}')
        tmp.replace(target)
        file_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest['evaluation_inputs_inherited'][str(target)] = file_hash
        manifest['datasets'].append({'path': item['path'], 'count': len(rows), 'content_sha256': actual, 'file_sha256': file_hash})
        print(f"VERIFIED {item['path']} ({len(rows)} rows)", flush=True)
    manifest['spec_sha256'] = hashlib.sha256(args.spec.read_bytes()).hexdigest()
    manifest['note'] = 'Canonical consumed fields and row order match the frozen reference; serialization/path hashes may differ.'
    tmp = receipt.with_suffix('.tmp')
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(receipt)
    print(f'All {len(manifest["datasets"])} data files verified: {receipt}')


if __name__ == '__main__':
    main()
