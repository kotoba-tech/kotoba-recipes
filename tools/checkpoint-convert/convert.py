import sys

sys.path.append("/bb/llm/gaf51275/llama/llama-recipes/src")

from transformers import (  # noqa: F401
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    default_data_collator,
)
from torch.distributed._shard.checkpoint import FileSystemReader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload  # type: ignore
from torch.distributed.fsdp import ShardingStrategy  # type: ignore
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType  # type: ignore
import torch
from typing import Any
import torch.distributed.checkpoint as dist_cp
import os
import argparse
import torch.distributed as torch_distributed
from llama_recipes.utils import fsdp_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from llama_recipes.utils.train_utils import clear_gpu_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Llama-Recipes convert")

    parser.add_argument("--hf-base-model-path", type=str, default=None, help="huggingface checkpoint path")
    parser.add_argument("--fsdp-checkpoint-path", type=str, default=None, help="FSDP checkpoint path")
    parser.add_argument("--checkpoint-output-path", type=str, default=None, help="output checkpoint path")

    args = parser.parse_args()
    return args


def main() -> None:
    # distributed setting
    global_rank = int(os.getenv("OMPI_COMM_WORLD_RANK", 0))
    local_rank = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", 0))
    world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", 1))

    os.environ["RANK"] = str(global_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch_distributed.init_process_group("nccl")
    print(f"torch.distributed.init rank={torch_distributed.get_rank()}", flush=True)

    args = parse_args()

    if torch_distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)

    # model definition
    if torch_distributed.get_rank() == 0:
        print("model loading start", flush=True)

    if torch_distributed.get_rank() == 0:
        model = LlamaForCausalLM.from_pretrained(
            args.hf_base_model_path,
            load_in_8bit=None,
            device_map=None,
            use_cache=None,
        )
    else:
        llama_config = LlamaConfig.from_pretrained(args.hf_base_model_path)
        llama_config.use_cache = False
        with torch.device("meta"):
            model = LlamaForCausalLM(llama_config)

    if torch_distributed.get_rank() == 0:
        print("model setup", flush=True)

    my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)

    fsdp_model = FSDP(
        module=model,  # type: ignore
        auto_wrap_policy=my_auto_wrapping_policy,
        cpu_offload=CPUOffload(offload_params=True),
        mixed_precision=None,  # bf16の場合は None にする
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        sync_module_states=True,
        param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False),  # type: ignore
    )
    if torch_distributed.get_rank() == 0:
        print("FSDP model setup done", flush=True)

    reader = FileSystemReader(
        args.fsdp_checkpoint_path
    )
    with FSDP.state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT):
        state_dict: dict[str, Any] = {
            "model": fsdp_model.state_dict(),
            # cannot load the optimizer state_dict together with the model state_dict
        }

        dist_cp.load_state_dict(  # type: ignore
            state_dict=state_dict,
            storage_reader=reader,
        )
        fsdp_model.load_state_dict(state_dict["model"])

    print("load fsdp checkpoint : rank = ", torch_distributed.get_rank(), flush=True)

    state_dict = get_model_state_dict(fsdp_model)
    print("get_model_state_dict : rank = ", torch_distributed.get_rank(), flush=True)

    if torch_distributed.get_rank() == 0:
        hf_model = LlamaForCausalLM.from_pretrained(
            args.hf_base_model_path,
            torch_dtype=torch.bfloat16,
        )
        # convert to hf checkpoint
        hf_model.load_state_dict(state_dict)  # type: ignore
        print("load_state_dict", flush=True)

        hf_model.save_pretrained(args.checkpoint_output_path)  # type: ignore

    torch_distributed.barrier()
    print("Done!: rank = ", torch_distributed.get_rank(), flush=True)


from torch.distributed.fsdp import FullStateDictConfig  # type: ignore


def get_model_state_dict(model: FSDP) -> dict[str, torch.Tensor]:
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        # state_dict() を呼ぶことで parameter を 保存してくれる
        state_dict = model.state_dict()

    return state_dict


if __name__ == "__main__":
    main()
