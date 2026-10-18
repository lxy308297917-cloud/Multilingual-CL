"""Exercise real Trainer restore helpers on saved TH states, without a GPU.

Validates serialization/runtime compatibility, not model parameter-name mapping,
CUDA RNG restoration, distributed execution, or resumed data-loader position.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
from types import SimpleNamespace

from resume_torch28_entry import enable_numpy_rng_compat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    metadata = enable_numpy_rng_compat()
    import numpy as np
    import torch
    from transformers import Trainer, get_linear_schedule_with_warmup
    from transformers.trainer import safe_globals
    from transformers.training_args import ParallelMode

    if torch.cuda.is_available():
        raise RuntimeError("Run with CUDA_VISIBLE_DEVICES empty; this is a CPU check")
    torch.set_num_threads(2)
    checkpoint = args.checkpoint.resolve()
    step = json.loads((checkpoint / "trainer_state.json").read_text())["global_step"]
    saved = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    parameters = {
        key: torch.nn.Parameter(torch.empty_like(value["exp_avg"], device="cpu"))
        for key, value in saved["state"].items()
    }
    groups = [{**group, "params": [parameters[key] for key in group["params"]]}
              for group in saved["param_groups"]]
    optimizer = torch.optim.AdamW(groups)
    del saved, groups
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, 100)
    shim = SimpleNamespace(
        is_deepspeed_enabled=False, is_fsdp_xla_v1_enabled=False,
        is_fsdp_enabled=False, optimizer=optimizer, lr_scheduler=scheduler,
        args=SimpleNamespace(world_size=1, device=torch.device("cpu")),
    )
    Trainer._load_optimizer_and_scheduler(shim, str(checkpoint))
    steps = sorted({int(value["step"].item()) for value in optimizer.state.values()})
    assert steps == [step], steps
    assert scheduler.last_epoch == step
    assert all(torch.isfinite(value[name]).all().item()
               for value in optimizer.state.values() for name in ["exp_avg", "exp_avg_sq"])
    rank_results = []
    for rank in range(4):
        shim.args = SimpleNamespace(world_size=4, process_index=rank,
                                    parallel_mode=ParallelMode.DISTRIBUTED)
        Trainer._load_rng_state(shim, str(checkpoint))
        # Match Trainer's context: its exit removes overlapping global entries.
        with safe_globals():
            state = torch.load(checkpoint / f"rng_state_{rank}.pth",
                               map_location="cpu", weights_only=True)
        assert random.getstate() == state["python"]
        actual = np.random.get_state()
        expected = state["numpy"]
        assert actual[0] == expected[0] and np.array_equal(actual[1], expected[1])
        assert actual[2:] == expected[2:]
        assert torch.equal(torch.random.get_rng_state(), state["cpu"])
        rank_results.append({"rank": rank, "python_numpy_cpu_rng_restored": True,
                             "cuda_rng_restored": False})
    result = {
        "observed_at": datetime.datetime.now().astimezone().isoformat(),
        "checkpoint": str(checkpoint), "global_step": step, "runtime": metadata,
        "versions": {p: importlib.metadata.version(p) for p in
                     ["torch", "transformers", "trl", "peft", "accelerate", "datasets", "numpy"]},
        "optimizer_state_count": len(optimizer.state), "optimizer_steps": steps,
        "optimizer_moments_finite": True, "scheduler_last_epoch": scheduler.last_epoch,
        "learning_rates": [group["lr"] for group in optimizer.param_groups],
        "rank_rng_results": rank_results,
        "bootstrap_sha256": hashlib.sha256(Path(__file__).with_name("resume_torch28_entry.py").read_bytes()).hexdigest(),
        "scope": __doc__.strip(),
        "distributed_model_resume_verified": False,
        "current_training_or_environment_modified": False,
    }
    with args.output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "global_step": step,
                      "optimizer_state_count": len(optimizer.state), "cpu_restore_helpers": "PASS"}))


if __name__ == "__main__":
    main()
