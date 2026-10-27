from pathlib import Path
import hashlib
import json
import time
import os

ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT/'outputs/benchmark_direction_20261026'
REPORT = PROJECT/'direction'
RUN = ROOT/'work/benchmark_direction_20261026/direction'
H = Path('/data0/users/lixiaoyang')
BASE = H/'models/Qwen2.5-1.5B-Instruct'
MAIN = ROOT/'outputs/Multilingual-CL-alignment-recovery-v3/experiments/representation_alignment_v1/data'
PYTHON = H/'ssu-migration/envs/cl/bin/python'
GROUPS = ('U', 'P', 'R', 'N', 'N_abs')

def account_limit(evaluation=False):
    """Human-authorized exception is local to this round's1.5B evaluation."""
    path = PROJECT/'RESOURCE_AUTHORIZATION.json'
    if evaluation and path.exists():
        auth = json.loads(path.read_text())
        if auth.get('active'):
            assert auth['max_account_GPUs_for_this_round_Qwen_1_5B_evaluation'] == 5
            assert auth['max_account_GPUs_for_training_and_training_preflight'] == 4
            return 5
    return 4

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n'); temp.replace(path)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()
