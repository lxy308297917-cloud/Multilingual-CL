"""Main SFT engine; calibration never contributes to protection-arm moments."""
from settings import *
import importlib.util
import random
import math
import shutil
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimizer import DirectionSparseAdamW
from calibration import collect

def learning_rate(step):
    assert 0 <= step <= 42465
    if step < 2124:
        return 3.857e-5*step/2124
    return 3.857e-5*.5*(1+math.cos(math.pi*(step-2124)/(42465-2124)))

def data():
    frozen = read(REPORT/'DATA_AND_BUDGET_FROZEN.json')
    manifest = frozen['main_manifest']
    assert sha(manifest['source']) == manifest['source_sha256']
    assert sha(RUN/'MAIN_PREFIX.json') == frozen['prefix_sha256']
    assert sha(RUN/'CALIBRATION_POOL.json') == frozen['calibration_pool_sha256']
    assert sha(RUN/'CALIBRATION_SCHEDULE.json') == frozen['calibration_schedule_sha256']
    source = H/'reproductions/sw_recipe_fft_20261010/train.py'
    spec = importlib.util.spec_from_file_location('direction_indexed_rows', source)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    stream = mod.IndexedRows(manifest['source'], read(RUN/'MAIN_PREFIX.json'))
    pool = {r['id']: r for r in read(RUN/'CALIBRATION_POOL.json')}
    schedule = {r['step']: r for r in read(RUN/'CALIBRATION_SCHEDULE.json')}
    return stream, pool, schedule

def setup(method):
    # Preserve the deterministic settings of the recent sparse engine; required
    # for a meaningful real-checkpoint continuation comparison.
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    assert os.environ['CUDA_VISIBLE_DEVICES'].startswith('GPU-') and torch.cuda.device_count() == 1
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, local_files_only=True,
                 torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    budget = read(REPORT/'DATA_AND_BUDGET_FROZEN.json')
    named = {n: p for n, p in model.named_parameters() if n in budget['quota']}
    assert len(named) == 196 and sum(p.numel() for p in named.values()) == 1310195712
    for n, p in model.named_parameters():
        p.requires_grad_(n in named)
    model.train()
    return model, tokenizer, named, DirectionSparseAdamW(named, budget['quota'], method)

def ce(model, row):
    device = next(model.parameters()).device
    ids = torch.tensor([row['input_ids']], device=device)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    targets = torch.tensor(row['labels'][1:], device=device)
    keep = targets != -100
    assert bool(keep.any())
    return torch.nn.functional.cross_entropy(logits[0, :-1][keep].float(), targets[keep])

def update(model, named, opt, batch, calibration_rows=None, next_cursor=None):
    assert len(batch) == 2
    model.train(); model.zero_grad(set_to_none=True)
    phases = {}; cal = None
    if opt.method != 'U' and opt.steps % 20 == 0:
        assert calibration_rows is not None and len(calibration_rows) == 8
        gradients, cal = collect(model, named, calibration_rows,
                                lambda row: ce(model, row), replay=opt.method == 'R')
        if opt.method != 'R':
            opt.calibrate(gradients, next_cursor)
            del gradients
        else:
            opt.calibration_cursor = next_cursor
            opt.calibrations += 1
        phases['calibration'] = cal['seconds']
    started = time.monotonic(); value = 0.
    for row in batch:
        loss = ce(model, row)/2
        value += float(loss.detach()); loss.backward(); del loss
    torch.cuda.synchronize(); phases['main'] = time.monotonic()-started
    norm = float(torch.nn.utils.clip_grad_norm_(list(named.values()), 1.))
    assert math.isfinite(value) and math.isfinite(norm)
    started = time.monotonic()
    stats = opt.step(learning_rate(opt.steps))
    torch.cuda.synchronize(); phases['optimizer'] = time.monotonic()-started
    stats.update(main_CE=value, combined_preclip_gradient_norm=norm,
                 calibration=cal, seconds_by_phase=phases)
    model.zero_grad(set_to_none=True)
    return stats

def rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())

def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])

def checkpoint(path, model, opt, counts, protocol_hash):
    path = Path(path)
    assert not path.exists()
    # Full BF16 model +FP32 moments/CPU calibration + packed masks, plus10GiB reserve.
    estimate = sum(p.numel()*p.element_size() for p in model.parameters())
    estimate += sum(v.numel()*v.element_size() for s in opt.state.values() for v in s.values())
    estimate += sum(v.numel()*v.element_size() for v in opt.english.values())
    estimate += sum(p.numel()/4 for p in opt.named.values())
    assert shutil.disk_usage(path.parent).free > estimate+10*1024**3
    path.mkdir()
    model.save_pretrained(path/'model', safe_serialization=True)
    state = dict(optimizer=opt.state_dict(), rng=rng(), counts=counts,
                 scheduler=dict(step=opt.steps, next_lr=learning_rate(opt.steps)),
                 protocol_sha256=protocol_hash)
    torch.save(state, path/'state.pt'); del state
    write(path/'RECOVERABLE.json', dict(step=opt.steps, files={str(p.relative_to(path)): sha(p)
                      for p in path.rglob('*') if p.is_file()}))

def load_checkpoint(path, model, opt, protocol_hash):
    from safetensors.torch import load_file
    path = Path(path); receipt = read(path/'RECOVERABLE.json')
    for rel, digest in receipt['files'].items():
        assert sha(path/rel) == digest
    state = torch.load(path/'state.pt', map_location='cpu', mmap=True, weights_only=False)
    assert state['protocol_sha256'] == protocol_hash
    missing = model.load_state_dict(load_file(str(path/'model/model.safetensors')), strict=False)
    assert missing.missing_keys == ['lm_head.weight'] and not missing.unexpected_keys
    opt.load_state_dict(state['optimizer']); restore_rng(state['rng'])
    assert state['scheduler'] == dict(step=opt.steps, next_lr=learning_rate(opt.steps))
    return state['counts']
