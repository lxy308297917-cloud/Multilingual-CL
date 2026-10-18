"""Launch the unchanged TH entry with explicit legacy NumPy RNG compatibility.

Used only by the separate Torch 2.8 resume environment. Never patches torch.load
or disables Transformers' Torch version check. A real distributed resume remains
unverified until a GPU run completes an update after loading the checkpoint.
"""
import json
from pathlib import Path
import runpy
import sys


def enable_numpy_rng_compat():
    import numpy as np
    import torch
    from transformers.utils.import_utils import check_torch_load_is_safe

    check_torch_load_is_safe()
    if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
        from numpy._core.multiarray import _reconstruct
    else:
        from numpy.core.multiarray import _reconstruct
    # Our NumPy 1.26 checkpoints encode the old module path and uint32 MT state.
    torch.serialization.add_safe_globals([
        (_reconstruct, "numpy.core.multiarray._reconstruct"),
        np.ndarray, np.dtype, type(np.dtype("uint32")),
    ])
    return {
        "torch": torch.__version__, "numpy": np.__version__,
        "numpy_rng_compat": "old reconstruct path and uint32 dtype allowlist",
        "weights_only_default_preserved": True,
    }


def main():
    metadata = enable_numpy_rng_compat()
    print(json.dumps({"resume_runtime": metadata}), flush=True)
    if sys.argv[1:] == ["--compat-check-only"]:
        return
    entry = Path("/data0/users/lixiaoyang/COPSD-ddp-teacher-unwrapped-20261018/multilingual_opsd_train.py")
    if not entry.is_file():
        raise FileNotFoundError(entry)
    sys.path.insert(0, str(entry.parent))
    sys.argv[0] = str(entry)
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
