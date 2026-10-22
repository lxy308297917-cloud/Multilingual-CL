"""One existing real long-update check with CPU call-time profiling of step()."""
import cProfile,io,json,pstats,sys
from pathlib import Path
from core import SparseProjectedAdamW
from stress_qwen import main

original_step=SparseProjectedAdamW.step

def profiled_step(self,*args,**kwargs):
    profile=cProfile.Profile();profile.enable()
    try:return original_step(self,*args,**kwargs)
    finally:
        profile.disable()
        output=Path(sys.argv[sys.argv.index('--output')+1])
        profile.dump_stats(str(output/'optimizer.prof'))
        stream=io.StringIO();pstats.Stats(profile,stream=stream).strip_dirs().sort_stats('tottime').print_stats(45)
        (output/'OPTIMIZER_PROFILE.txt').write_text(stream.getvalue())
        print(stream.getvalue(),flush=True)

SparseProjectedAdamW.step=profiled_step
if __name__=='__main__':main()
