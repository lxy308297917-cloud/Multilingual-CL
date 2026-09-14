"""CPU-only data identity tests; no model or network calls."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

p = Path(__file__).resolve().parents[1] / 'scripts/src/prepare_baseline_eval_data.py'
spec = importlib.util.spec_from_file_location('public_data', p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class PublicDataTests(unittest.TestCase):
    def test_identity_includes_order_and_consumed_content(self):
        a = [{'x': 'a'}, {'x': 'b'}]
        self.assertNotEqual(m.digest(a, ['x']), m.digest(a[::-1], ['x']))
        self.assertNotEqual(m.digest(a, ['x']), m.digest([{'x': 'c'}, a[1]], ['x']))
        self.assertEqual(m.digest(a, ['x']), m.digest([dict(r, extra=1) for r in a], ['x']))

    def test_frozen_ids_do_not_resample(self):
        rows = [{'id': 'a'}, {'id': 'b'}]
        item = {'selection': 'frozen_ids', 'ids': ['b', 'a']}
        self.assertEqual(m.select_rows(item, rows), rows[::-1])
        with self.assertRaises(ValueError):
            m.select_rows(item, rows + rows)
        with self.assertRaises(KeyError):
            m.select_rows(dict(item, ids=['missing']), rows)

    def test_jsonl_unicode_line_separators_are_text(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'test.jsonl'
            rows = [{'x': 'a\u2028b\u0085c'}]
            p.write_text(json.dumps(rows[0], ensure_ascii=False) + '\n')
            self.assertEqual(m.read_rows(p), rows)

    def test_reject_mutable_source_revision(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                m.download({'revision': 'main'}, Path(d))


if __name__ == '__main__':
    unittest.main()
