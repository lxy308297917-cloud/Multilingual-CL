"""Grade censored first2048 tokens of the same completed long diagnostic trajectories."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
from evaluate_vllm import official_api


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--long-report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    receipt = read(a.long_report)
    assert receipt['protocol'] == 'privileged_teacher_conditioning_length8192_v1'
    assert receipt['queue_phase'] == 'completed' and receipt['exit_code'] == 0
    folder = a.root / 'logs/copsd_official_10lang/teacher_context'
    long = folder / 'te_conditioning_length8192_dev100_20261021_v1'
    short = folder / 'te_conditioning_dev100_20261021_v2'
    lc, sc = [read(x / 'CONFIG.PRIVATE.json') for x in [long, short]]
    assert lc['max_new_tokens'] == 8192 and sc['max_new_tokens'] == 2048
    for k in ['prompts_sha256', 'language', 'model', 'source', 'collator_sha256', 'language_config_sha256', 'seed', 'temperature', 'top_p', 'top_k']:
        assert lc[k] == sc[k], k
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(lc['model'], local_files_only=True)
    source = Path(lc['source']) / 'polymath_eval/evaluate_math.py'
    api = official_api(source, {'LLM': None, 'SamplingParams': None})
    result = {}
    for condition in ['student', 'teacher']:
        path = long / (condition + '_ITEMS.PRIVATE.jsonl')
        assert sha(path) == receipt['source_records_sha256'][condition]
        rr = [json.loads(s) for s in path.read_text().splitlines()]
        old = [json.loads(s) for s in (short / (condition + '_ITEMS.PRIVATE.jsonl')).read_text().splitlines()]
        assert len(rr) == len(old) == 100
        censored, exact, boxes = [], 0, 0
        for r, previous in zip(rr, old):
            assert all(r[k] == previous[k] for k in ['id', 'answer', 'seed'])
            assert tokenizer.decode(r['token_ids'], skip_special_tokens=False) == r['text']
            text = tokenizer.decode(r['token_ids'][:2048], skip_special_tokens=False)
            answer = api['extract_boxed_answer'](text)
            correct = bool(api['grade_answer'](answer, r['answer']))
            if len(r['token_ids']) <= 2048: assert correct == bool(r['correct'])
            censored.append(correct); boxes += answer is not None
            length = min(2048, len(previous['token_ids']))
            exact += r['token_ids'][:length] == previous['token_ids'][:length]
        full = [bool(r['correct']) for r in rr]
        result[condition] = {'same_long_trajectory_correct_first2048': sum(censored),
                             'same_long_trajectory_correct_full8192': sum(full),
                             'gained_correct_after2048': sum(not x and y for x, y in zip(censored, full)),
                             'lost_correct_after2048': sum(x and not y for x, y in zip(censored, full)),
                             'boxed_answer_available_first2048': boxes,
                             'separate_short_run_correct': sum(r['correct'] for r in old),
                             'separate_short_vs_long_prefix_exact_matches': exact}
    out = {'observed_at': datetime.datetime.now().isoformat(), 'items_each': 100,
           'same_trajectory_answer_availability': result, 'official_scorer_source_sha256': sha(source),
           'long_report_sha256': sha(a.long_report), 'analysis_source_sha256': sha(Path(__file__)),
           'new_training_or_inference': False,
           'scope': 'CPU censoring only, not additional generation, model quality ranking or SFT labels. Earlier separate2048 runs may differ in generated prefixes despite equal per-ID seeds; same-long-trajectory censoring isolates later answer availability without proving intrinsic mathematical gain or on-policy supervision quality.'}
    with a.output.open('x') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
