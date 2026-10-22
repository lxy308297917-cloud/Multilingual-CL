"""Launch the installed TRL server with an independently importable audited worker."""
import trl.scripts.vllm_serve as released
from trl_server_worker32 import llm_worker


if __name__ == '__main__':
    parser = released.make_parser()
    (args,) = parser.parse_args_and_config()
    released.llm_worker = llm_worker
    released.main(args)
