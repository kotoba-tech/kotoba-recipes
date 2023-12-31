from transformers import LlamaConfig, LlamaForCausalLM, MistralForCausalLM
from llama_recipes.configs import train_config
from typing import Type
from llama_recipes.utils.distributed import is_rank_0
import torch


def get_model(train_config: Type[train_config], use_cache: bool = False) -> LlamaForCausalLM | MistralForCausalLM:
    """return CausalLM model

    Args:
        train_config (Type[train_config]):
        use_cache (bool, optional):

    Raises:
        NotImplementedError: currently only supports LlamaForCausalLM and MistralForCausalLM

    Returns:
        LlamaForCausalLM | MistralForCausalLM: PyTorch model
    """
    if "Llama" in train_config.model_name:
        if train_config.enable_fsdp and train_config.low_cpu_fsdp:
            """
            for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
            this avoids cpu oom when loading large models like llama 70B, in which case
            model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some communications
            overhead.
            """
            if is_rank_0():
                model = LlamaForCausalLM.from_pretrained(
                    train_config.model_name,
                    load_in_8bit=True if train_config.quantization else None,
                    device_map="auto" if train_config.quantization else None,
                    use_cache=use_cache,
                )
            else:
                llama_config = LlamaConfig.from_pretrained(train_config.model_name)
                llama_config.use_cache = use_cache
                with torch.device("meta"):
                    model = LlamaForCausalLM(llama_config)

        else:
            model = LlamaForCausalLM.from_pretrained(
                train_config.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
            )

        return model  # type: ignore

    elif "Mistral" in train_config.model_name:
        mistral_max_length: int = 4096
        sliding_window: int = 4096

        model = MistralForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache=use_cache,
            sliding_window=sliding_window,
            max_position_embeddings=mistral_max_length,
            use_flash_attention_2=True,
        )

        return model  # type: ignore

    else:
        raise NotImplementedError("model not implemented")
