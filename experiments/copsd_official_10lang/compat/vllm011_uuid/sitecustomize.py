"""Process-local vLLM 0.11 NVML mapping for UUID-based CUDA visibility.

Activate only for the isolated vLLM probe via PYTHONPATH and the explicit flag.
Keep CUDA_VISIBLE_DEVICES unchanged; CUDA still selects the queue's UUIDs.
Propagates to vLLM's Python inspection/worker subprocesses without editing its
installed package, the running environment, or any training algorithm.
"""
import os
import sys


if os.environ.get("COPSD_VLLM_UUID_COMPAT") == "1":
    try:
        from vllm.platforms.interface import Platform
        original = Platform.device_id_to_physical_device_id.__func__

        def device_id_to_physical_device_id(cls, device_id):
            values = os.environ.get(cls.device_control_env_var, "").split(",")
            if values and values[0]:
                identifier = values[device_id]
                if identifier.startswith("GPU-"):
                    from vllm.third_party import pynvml
                    pynvml.nvmlInit()
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByUUID(identifier)
                        return pynvml.nvmlDeviceGetIndex(handle)
                    finally:
                        pynvml.nvmlShutdown()
            return original(cls, device_id)

        Platform.device_id_to_physical_device_id = classmethod(device_id_to_physical_device_id)
        print("[COPSD UUID compat] vLLM NVML mapping enabled; CUDA visibility unchanged", file=sys.stderr)
    except Exception as error:
        print("[COPSD UUID compat] refusing startup: " + repr(error), file=sys.stderr, flush=True)
        os._exit(86)
