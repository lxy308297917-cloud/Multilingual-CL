#!/usr/bin/env python
import argparse
import os
import sys

import main_bf16
from utils.mofo import MoFOTrainer


def main():
    # Transformers blocks optimizer.pt loading with torch<2.6 because pickle
    # checkpoints may execute code. Allow bypassing that check only for an
    # explicitly trusted, locally generated checkpoint so exact optimizer and
    # scheduler state can be restored without weakening other entry points.
    if os.environ.get("TRUST_LOCAL_MOFO_CHECKPOINT", "0") == "1":
        import transformers.trainer as transformers_trainer

        transformers_trainer.check_torch_load_is_safe = lambda: None
        print("[MoFO] Trusting local optimizer checkpoint for exact resume")

    mofo_parser = argparse.ArgumentParser(add_help=False)
    mofo_parser.add_argument("--mofo_update_fraction", type=float, default=0.15)
    mofo_args, remaining = mofo_parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    class ConfiguredMoFOTrainer(MoFOTrainer):
        mofo_update_fraction = mofo_args.mofo_update_fraction

    main_bf16.Trainer = ConfiguredMoFOTrainer
    parser = main_bf16.CustomArgumentParser()
    args, training_args = parser.parse_args()
    print(
        f"[MoFO] Per-parameter-tensor momentum filtering enabled: "
        f"alpha={mofo_args.mofo_update_fraction:.2%}"
    )
    main_bf16.main(args, training_args)


if __name__ == "__main__":
    main()
