"""One frozen4293-update arm. Launch only after technical and scientific gates."""
from settings import *
import argparse
import traceback
from resources import GPUContext, usage_seconds

ap = argparse.ArgumentParser()
ap.add_argument('--uuid', required=True)
ap.add_argument('--method', choices=GROUPS, required=True)
ap.add_argument('--run-id', default='first')
ap.add_argument('--resume', type=Path)
a = ap.parse_args()
assert a.run_id.replace('_', '').replace('-', '').isalnum()
protocol = REPORT/'PROTOCOL_FROZEN.json'
assert protocol.exists(), 'No scientific training before joint protocol/cost freeze'
proto = read(protocol)
assert sha(PROJECT/'resource_guard.py') == proto['shared_resource_guard_sha256']
assert proto['training_launch_authorized_after_bridge_review']
assert read(REPORT/'REAL_QWEN_PREFLIGHT.json')['passed']
assert (PROJECT/'bridge/COMPLETION_VERIFIED.json').exists()
for rel, digest in proto['code_sha256'].items():
    assert sha(REPORT/rel) == digest
out = H/'models/checkpoint/direction_risk_20261026'/f'{a.method}_{a.run_id}'
logs = H/'logs/direction_risk_20261026'/f'{a.method}_{a.run_id}'
assert not out.exists() and not logs.exists(), 'Independent output required; preserve previous attempt'
try:
    with GPUContext(a.uuid, a.method+'_'+a.run_id) as context:
        import torch
        from torch.utils.tensorboard import SummaryWriter
        import engine
        out.mkdir(parents=True); logs.mkdir(parents=True)
        write(out/'RUN_IDENTITY.json', dict(method=a.method, run_id=a.run_id,
              protocol_sha256=sha(protocol), uuid=a.uuid, pid=os.getpid(),
              output_real_path=str(out.resolve()), resume=str(a.resume) if a.resume else None))
        model, tokenizer, named, opt = engine.setup(a.method)
        rows, pool, schedule = engine.data()
        counts = dict(main_input=0, main_supervised=0, calibration_input=0,
                      calibration_supervised=0, replay_input=0, replay_supervised=0,
                      generated_tokens=0, cumulative_actual_path=0.)
        if a.resume:
            counts = engine.load_checkpoint(a.resume, model, opt, sha(protocol))
        writer = SummaryWriter(str(logs/'tensorboard'))
        with (logs/'steps.jsonl').open('x') as stream:
            while opt.steps < 4293:
                i = opt.steps
                batch = [rows[2*i], rows[2*i+1]]
                event = schedule.get(i)
                cal = [pool[qid] for qid in event['ids']] if event else None
                stats = engine.update(model, named, opt, batch, cal,
                                      event['next_cursor'] if event else None)
                counts['main_input'] += sum(len(r['input_ids']) for r in batch)
                counts['main_supervised'] += sum(sum(y != -100 for y in r['labels'][1:]) for r in batch)
                counts['cumulative_actual_path'] += stats['actual_norm']
                if stats['calibration']:
                    key = 'replay' if a.method == 'R' else 'calibration'
                    counts[key+'_input'] += stats['calibration']['input_tokens']
                    counts[key+'_supervised'] += stats['calibration']['supervised_tokens']
                stream.write(json.dumps(dict(stats, counts=counts), ensure_ascii=False)+'\n')
                if opt.steps % 10 == 0:
                    stream.flush()
                    for key in ('main_CE', 'actual_norm', 'candidate_norm', 'nonzero_write', 'mask_union', 'combined_preclip_gradient_norm'):
                        writer.add_scalar(key, stats[key], opt.steps)
                    write(out/'PROGRESS.json', dict(step=opt.steps, counts=counts,
                          accounted_stage_GPU_hours=usage_seconds()[0]/3600, updated_unix=time.time()))
                if opt.steps in (2146, 4293):
                    engine.checkpoint(out/f'checkpoint_{opt.steps:06d}', model, opt, dict(counts), sha(protocol))
            writer.close()
        assert counts['main_input'] == 1536351 and counts['main_supervised'] == 599469
        assert opt.calibrations == (0 if a.method == 'U' else 215)
        exposure = read(REPORT/'EXPOSURE_FROZEN.json')
        assert counts['generated_tokens'] == 0
        if a.method == 'U':
            assert counts['calibration_input'] == counts['replay_input'] == 0
        else:
            key = 'replay' if a.method == 'R' else 'calibration'
            assert counts[key+'_input'] == exposure['auxiliary_input']
            assert counts[key+'_supervised'] == exposure['auxiliary_supervised']
            other = 'calibration' if a.method == 'R' else 'replay'
            assert counts[other+'_input'] == counts[other+'_supervised'] == 0
        # Per-matrix CPU comparison avoids retaining a second full model on GPU.
        from safetensors import safe_open
        net = {}; total_sq = 0.
        with safe_open(str(BASE/'model.safetensors'), framework='pt', device='cpu') as base:
            frozen_count = 0
            for n, p in model.named_parameters():
                if n not in named:
                    assert not p.requires_grad
                    assert torch.equal(p.detach().cpu(), base.get_tensor(n).to(p.dtype)), 'Frozen parameter changed: '+n
                    frozen_count += p.numel()
            for n, p in named.items():
                delta = p.detach().float().cpu()-base.get_tensor(n).float()
                sq = float(delta.square().sum()); total_sq += sq
                key = 'Attention' if '.self_attn.' in n else 'MLP'
                net[key] = net.get(key, 0.)+sq
        final = out/'checkpoint_004293/model/model.safetensors'
        write(out/'TRAINING_DONE.json', dict(method=a.method, steps=opt.steps, counts=counts,
              final_model_sha256=sha(final), final_net_displacement=total_sq**.5,
              module_net_displacement={k: v**.5 for k, v in net.items()},
              frozen_parameter_elements_verified=frozen_count,
              frozen_parameters_equal_initial_Base_cast=True,
              calibrations=opt.calibrations, calibration_cursor=opt.calibration_cursor,
              peak_allocated_bytes=torch.cuda.max_memory_allocated(),
              protocol_sha256=sha(protocol), finished_unix=time.time()))
except BaseException as e:
    if isinstance(e, SystemExit) and e.code == 75:
        raise
    write(RUN/f'FAILED_{a.method}_{a.run_id}.json', dict(error=str(e), traceback=traceback.format_exc(),
          preserved_output=str(out), failed_unix=time.time()))
    raise
