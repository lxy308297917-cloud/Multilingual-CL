"""Shared account reservation and GPU-context accounting for this one-shot stage."""
from settings import *
import sys
sys.path.insert(0,str(ROOT/'outputs/benchmark_direction_20261026'))
from resource_guard import all_queue_reservations
import fcntl
import signal
import threading
import subprocess

def usage_seconds():
    total = 0.; live = 0
    for p in RUN.glob('GPU_START_*.json'):
        row = read(p); end = RUN/p.name.replace('GPU_START_', 'GPU_END_', 1)
        total += (read(end)['finished_unix'] if end.exists() else time.time())-row['started_unix']
        live += not end.exists()
    return total, live

class GPUContext:
    def __init__(self, uuid, label, evaluation=False):
        self.uuid, self.label = uuid, label
        self.evaluation = evaluation
        self.stop = threading.Event(); self.physical = []; self.started = None

    def __enter__(self):
        import sys
        sys.path.insert(0, str(H/'gpu-queue'))
        import gpu_queue as q
        with (H/'gpu-queue/reservation.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for _ in range(2):
                snap = q.snapshot(); reserved = all_queue_reservations(q)
                assigned = q.user_gpu_assignments(snap, reserved)
                limit = 4
                if len(assigned) >= limit or self.uuid in reserved or not q.idle(snap[self.uuid], {'max_idle_memory_mib': 512, 'max_idle_util_percent': 5}):
                    write(RUN/f'RESOURCE_WAIT_{self.label}_{time.time_ns()}.json', dict(
                          uuid=self.uuid, assigned=sorted(assigned), started_GPU_context=False, unix=time.time()))
                    raise SystemExit(75)
                time.sleep(1)
            os.environ['CUDA_VISIBLE_DEVICES'] = self.uuid
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            import torch
            torch.cuda.init(); probe = torch.empty(1, device='cuda'); torch.cuda.synchronize(); del probe
            self.started = time.time()
            assert not (RUN/f'GPU_START_{self.label}.json').exists()
            write(RUN/f'GPU_START_{self.label}.json', dict(started_unix=self.started,
                  pid=os.getpid(), uuid=self.uuid, assigned_before=sorted(assigned),
                  account_gpu_limit=limit, evaluation_only_exception=self.evaluation and limit == 5))
        def monitor():
            while not self.stop.wait(5):
                try:
                    self.physical.append(int(subprocess.check_output(['nvidia-smi', '--id='+self.uuid,
                        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True, timeout=10).strip()))
                except (OSError, ValueError, subprocess.SubprocessError):
                    pass
        threading.Thread(target=monitor, daemon=True).start()
        return self

    def __exit__(self, kind, error, tb):
        self.stop.set()
        if self.started is not None:
            write(RUN/f'GPU_END_{self.label}.json', dict(finished_unix=time.time(),
                  wall_seconds=time.time()-self.started, success=kind is None,
                  observed_physical_peak_MiB=max(self.physical) if self.physical else None,
                  error=str(error) if error else None))
