"""Use the installed TRL worker; audit tensors received and loaded by its loader.

The worker loop is the installed TRL0.26 llm_worker, with only the extension
class path changed. Model weights are received by the original NCCL mechanism.
"""
import hashlib
import json
import os
from pathlib import Path
import inspect
import trl.scripts.vllm_serve as _released
from trl.scripts.vllm_serve import WeightSyncWorkerExtension as ReleasedExtension

ORIGINAL_WORKER_SOURCE = inspect.getsource(_released.llm_worker)


class AuditedWeightSync(ReleasedExtension):
    def update_named_param(self, name, dtype, shape):
        model = self.model_runner.model
        original = model.load_weights
        def loaded(weights):
            values = list(weights)
            result = original(weights=values)
            path = Path(os.environ['COPSD_SERVER_RECEIPTS'])
            with path.open('a') as stream:
                for key, tensor in values:
                    digest = hashlib.sha256(tensor.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()
                    stream.write(json.dumps({'name': key, 'shape': list(tensor.shape),
                        'dtype': str(tensor.dtype), 'sha256': digest, 'original_loader_returned': True})+'\n')
            return result
        model.load_weights = loaded
        try:
            return super().update_named_param(name, dtype, shape)
        finally:
            model.load_weights = original


def llm_worker(script_args, data_parallel_rank, master_port, connection):
    # Retain installed worker behavior; replace only its configured extension.
    import ast
    import trl.scripts.vllm_serve as released
    source = ORIGINAL_WORKER_SOURCE
    assert source.count('trl.scripts.vllm_serve.WeightSyncWorkerExtension') == 1
    source = source.replace('trl.scripts.vllm_serve.WeightSyncWorkerExtension',
        'trl_server_audited_worker.AuditedWeightSync')
    namespace = dict(vars(released))
    exec(compile(ast.parse(source), 'installed_TRL_worker_with_audit_extension', 'exec'), namespace)
    namespace['llm_worker'](script_args, data_parallel_rank, master_port, connection)
