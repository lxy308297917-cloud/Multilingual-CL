"""Read-only evaluation server with single-GPU canonical LoRA initialization."""
import trl.scripts.vllm_serve as released
from trl_server_eval_direct_lora_worker32 import llm_worker


if __name__=='__main__':
    parser=released.make_parser();(args,)=parser.parse_args_and_config()
    assert args.data_parallel_size==args.tensor_parallel_size==1
    released.llm_worker=llm_worker;released.main(args)
