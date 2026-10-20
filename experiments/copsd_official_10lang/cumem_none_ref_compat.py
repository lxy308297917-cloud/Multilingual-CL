"""Process-local repair for one verified broken abi3 binary on CPython <=3.11.

Never edit the installed package or apply to an unknown/correctly built binary.
Known functions return borrowed None where CPython requires an owned reference.
"""
import ctypes
import functools
import hashlib
import platform
import sys
from pathlib import Path

incref=ctypes.pythonapi.Py_IncRef
incref.argtypes=[ctypes.py_object]
incref.restype=None


def compensate_borrowed_none(function):
    @functools.wraps(function)
    def call(*args,**kwargs):
        result=function(*args,**kwargs)
        if result is not None:
            raise RuntimeError('Unexpected native return; refusing compatibility repair')
        # Only after successful native return. Exceptions get no extra reference.
        incref(result)
        return result
    return call


def install(expected_binary_sha256):
    assert platform.python_implementation()=='CPython' and sys.version_info[:2]==(3,10)
    assert len(expected_binary_sha256)==64
    from vllm import cumem_allocator as extension
    import vllm.device_allocator.cumem as allocator
    binary=Path(extension.__file__)
    assert hashlib.sha256(binary.read_bytes()).hexdigest()==expected_binary_sha256
    assert not getattr(allocator,'_copsd_none_reference_repaired',False)
    for name in ['python_create_and_map','python_unmap_and_release']:
        raw=getattr(extension,name)
        assert getattr(allocator,name) is raw, 'Unexpected pre-existing allocator patch'
        wrapped=compensate_borrowed_none(raw)
        setattr(extension,name,wrapped)
        setattr(allocator,name,wrapped)
    allocator._copsd_none_reference_repaired=True
    return {'process_local_only':True,'installed_binary_unchanged':True,
            'CPython':'3.10','known_binary_identity_verified':True,
            'compensated_native_functions':2,
            'scope':'None return-reference ownership only. No CUDA kernel, model, loss, data or sampling change.'}
