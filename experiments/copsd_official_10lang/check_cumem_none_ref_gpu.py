"""Tiny isolated CUDA allocator test; no model or optimizer is loaded."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('GPU_QUEUE_GPU_COUNT')=='1'
    assert os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-')
    assert not a.output.exists()
    lock=json.loads(a.lock.read_text());directory=Path(__file__).parent
    for name,expected in lock['code_sha256'].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==expected,name
    import torch
    import vllm.device_allocator.cumem as allocator
    from cumem_none_ref_compat import install
    torch.cuda.init()
    pool=allocator.CuMemAllocator.get_instance()
    with pool.use_memory_pool(tag='probe'):
        tensor=torch.empty(262144,device='cuda',dtype=torch.float32)
    tensor.fill_(7);torch.cuda.synchronize()
    handles=[d.handle for d in pool.pointer_to_data.values()
             if d.handle[2]<=tensor.data_ptr()<d.handle[2]+d.handle[1]]
    assert len(handles)==1
    handle=handles[0]
    def cycles(count):
        for _ in range(count):
            allocator.python_unmap_and_release(*handle)
            allocator.python_create_and_map(*handle)
    before=sys.getrefcount(None);cycles(5);raw_delta=sys.getrefcount(None)-before
    assert raw_delta==-10,raw_delta
    repair=install(lock['binary_sha256'])
    before=sys.getrefcount(None);cycles(100);fixed_delta=sys.getrefcount(None)-before
    assert fixed_delta==0,fixed_delta
    tensor.fill_(7);torch.cuda.synchronize()
    for _ in range(3):
        pool.sleep(offload_tags=('probe',));pool.wake_up()
        assert bool((tensor==7).all())
    del tensor
    torch.cuda.synchronize();torch.cuda.empty_cache()
    from vllm_ordered_shutdown import release_pools
    release_pools(pool)
    assert not pool.pointer_to_data
    result={'actual_vllm_version':'0.11.0','CPython':'3.10',
        'raw_five_native_remap_cycles_None_reference_delta':raw_delta,
        'repaired100_native_remap_cycles_None_reference_delta':fixed_delta,
        'actual_allocator_sleep_wake_preserved_tensor_values_cycles':3,
        'model_loaded':False,'optimizer_or_training_run':False,
        'ordered_pool_release_completed':True,
        'tensor_elements':262144,'tensor_dtype':'float32','compatibility':repair,
        'scope':'Actual isolated GPU allocator verification; does not modify installed binary or running processes. '
                'Does not establish full OPD training stability or mathematical effectiveness.'}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
