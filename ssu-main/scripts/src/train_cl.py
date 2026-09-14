#!/usr/bin/env python3
"""Stable public entry point for multilingual CL training.

The implementation remains in the legacy module until the two controllers
launched from that path finish.  This indirection can then be removed without
changing any experiment command.
"""

import json
from pathlib import Path
import runpy
import sys


def _config_path():
    if "--config" not in sys.argv:
        return None
    index = sys.argv.index("--config")
    return Path(sys.argv[index + 1])


def main():
    config = _config_path()
    if config is not None and json.loads(config.read_text()).get("experiment_family") == "ig_baseline_train_v1":
        target = Path(__file__).with_name("run_baseline_train.py")
        sys.argv = [str(target), *sys.argv[1:]]
        runpy.run_path(str(target), run_name="__main__")
        return
    if "--config" in sys.argv:
        path = Path(sys.argv[sys.argv.index("--config") + 1])
        if json.loads(path.read_text()).get("experiment_family") == "ig_lape_sparse_v1":
            target = Path(__file__).with_name("run_lape_ig.py")
            rest = sys.argv[1:]
            if rest and rest[0] in ("phase-a", "base", "phase-b"):
                rest = rest[1:]
            action = "train"
            if "--smoke" in rest:
                rest.remove("--smoke")
                action = "train-smoke"
            sys.argv = [str(target), action, *rest]
            runpy.run_path(str(target), run_name="__main__")
            return

    config = _config_path()
    if config is not None and json.loads(config.read_text()).get("experiment_family") in {"ig_aligned15k_plnd", "ig_aligned15k_plnd_protect"}:
        target = Path(__file__).with_name("run_plnd_ig.py")
        sys.argv = [str(target), "train", *sys.argv[1:]]
        runpy.run_path(str(target), run_name="__main__")
        return
    from run_multilingual_cl_training import main as legacy_main
    legacy_main()


if __name__ == "__main__":
    main()
