"""Retain TRL worker loop with a32-request resource limit and audited loader."""
import ast
from trl_server_audited_worker import ORIGINAL_WORKER_SOURCE


def worker_source():
    source=ORIGINAL_WORKER_SOURCE
    assert source.count('worker_extension_cls="trl.scripts.vllm_serve.WeightSyncWorkerExtension"')==1
    assert source.count('llm = LLM(')==1 and 'max_num_seqs=' not in source
    source=source.replace('worker_extension_cls="trl.scripts.vllm_serve.WeightSyncWorkerExtension"',
        'worker_extension_cls="trl_server_audited_worker.AuditedWeightSync", max_num_seqs=32')
    return source


def llm_worker(script_args,data_parallel_rank,master_port,connection):
    import trl.scripts.vllm_serve as released
    namespace=dict(vars(released))
    exec(compile(ast.parse(worker_source()),'installed_TRL_worker_with_audit_and_seqs32','exec'),namespace)
    namespace['llm_worker'](script_args,data_parallel_rank,master_port,connection)
