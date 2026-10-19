"""Release vLLM CUDA pools while their allocator owners are still alive.

Compatibility adaptation for vLLM 0.11; no training or sampling changes.
Upstream reference: vllm/device_allocator/cumem.py, release_pools on main.
https://github.com/vllm-project/vllm/blob/main/vllm/device_allocator/cumem.py
The first local probe aborted in MemPool destruction during Py_FinalizeEx.
This adaptation requires a real GPU run and normal exit to be accepted.
"""
import gc


def release_pools(allocator):
    entries = list(allocator.allocator_and_pools.values())
    owners = [entry[1] for entry in entries]
    pools = [entry[0] for entry in entries]
    allocator.allocator_and_pools.clear()
    entries.clear()
    pools.clear()
    gc.collect()
    owners.clear()


def shutdown(trainer, wake):
    import torch
    from vllm.device_allocator.cumem import CuMemAllocator
    from vllm.distributed.parallel_state import destroy_model_parallel

    engine = trainer.vllm_engine
    # Freeing a sleeping allocation must not access an unmapped CUDA address.
    wake(engine)
    torch.cuda.synchronize()
    engine.llm_engine.model_executor.shutdown()
    trainer.vllm_engine = None
    del engine
    gc.collect()
    release_pools(CuMemAllocator.get_instance())
    torch.cuda.empty_cache()
    destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
