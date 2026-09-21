#!/usr/bin/env python3
"""One command surface for Phase-A, Base, and Phase-B LightEval runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import os
import runpy
import sys


ROOT = Path(__file__).resolve().parent
COMMANDS = {
    "phase-a": ("run_multilingual_cl.py", ["stage-a-eval"]),
    "base": ("run_multilingual_cl_base_eval.py", []),
    "phase-b": ("run_multilingual_cl_phase_b_eval.py", []),
}


def main() -> None:
    if "--config" in sys.argv:
        path = Path(sys.argv[sys.argv.index("--config") + 1])
        if json.loads(path.read_text()).get("experiment_family") == "sw_d2_sparse_projection_v1":
            target = ROOT.parents[1] / "experiments/sw_d2_sparse_projection_v1/evaluate_method.py"
            previous = os.environ.get("SW_D2_METHOD_UNIFIED_EVAL_ENTRY")
            os.environ["SW_D2_METHOD_UNIFIED_EVAL_ENTRY"] = "1"
            try:
                sys.argv = [str(target), *sys.argv[1:]]
                runpy.run_path(str(target), run_name="__main__")
            finally:
                if previous is None: os.environ.pop("SW_D2_METHOD_UNIFIED_EVAL_ENTRY", None)
                else: os.environ["SW_D2_METHOD_UNIFIED_EVAL_ENTRY"] = previous
            return
    if "--config" in sys.argv:
        path = Path(sys.argv[sys.argv.index("--config") + 1])
        if json.loads(path.read_text()).get("experiment_family") == "sw_data_recipes_v4":
            target = ROOT / "sw_recipes_v4" / "evaluate.py"
            sys.path.insert(0, str(target.parent))
            sys.argv = [str(target), *sys.argv[1:]]
            runpy.run_path(str(target), run_name="__main__")
            return
    if "--config" in sys.argv:
        path = Path(sys.argv[sys.argv.index("--config") + 1])
        if json.loads(path.read_text()).get("experiment_family") == "sw_data_recipes_v3":
            target = ROOT / "sw_recipes_v3" / "evaluate.py"
            sys.path.insert(0, str(target.parent))
            sys.argv = [str(target), *sys.argv[1:]]
            runpy.run_path(str(target), run_name="__main__")
            return
    if "--config" in sys.argv:
        path = Path(sys.argv[sys.argv.index("--config") + 1])
        if json.loads(path.read_text()).get("experiment_family") == "ig_baseline_eval_v1":
            target = ROOT / "run_baseline_eval.py"
            sys.argv = [str(target), *sys.argv[1:]]
            runpy.run_path(str(target), run_name="__main__")
            return
    if "--config" in sys.argv:
        path = Path(sys.argv[sys.argv.index("--config") + 1])
        if json.loads(path.read_text()).get("experiment_family") == "ig_baseline_decoding_v1":
            target = ROOT / "run_ig_decoding_diagnostic.py"
            rest = sys.argv[1:]
            if rest and rest[0] in COMMANDS:
                rest = rest[1:]
            sys.argv = [str(target), *rest]
            runpy.run_path(str(target), run_name="__main__")
            return
    if "--config" in sys.argv:
        path = Path(sys.argv[sys.argv.index("--config") + 1])
        if json.loads(path.read_text()).get("experiment_family") == "ig_lape_sparse_v1":
            target = Path(__file__).with_name("run_lape_ig.py")
            rest = sys.argv[1:]
            if rest and rest[0] in ("phase-a", "base", "phase-b"):
                rest = rest[1:]
            action = "evaluate"
            if "--smoke" in rest:
                rest.remove("--smoke")
                action = "eval-smoke"
            sys.argv = [str(target), action, *rest]
            runpy.run_path(str(target), run_name="__main__")
            return

    if "--config" in sys.argv:
        config_index = sys.argv.index("--config")
        config = Path(sys.argv[config_index + 1])
        if json.loads(config.read_text()).get("experiment_family") in {"ig_aligned15k_plnd", "ig_aligned15k_plnd_protect"}:
            target = ROOT / "run_plnd_ig.py"
            remainder = sys.argv[1:]
            if remainder and remainder[0] in COMMANDS:
                remainder = remainder[1:]
            sys.argv = [str(target), "evaluate", *remainder]
            runpy.run_path(str(target), run_name="__main__")
            return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", choices=COMMANDS)
    args, remainder = parser.parse_known_args()
    filename, injected = COMMANDS[args.scope]
    target = ROOT / filename
    sys.argv = [str(target), *injected, *remainder]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
