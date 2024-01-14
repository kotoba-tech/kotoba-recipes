# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass
from typing import Optional
from torch.distributed.fsdp import ShardingStrategy  # type: ignore
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType  # type: ignore
import torch


@dataclass
class train_config:
    model_name: str = "kotoba-recipes-model-name"
    tokenizer_name: str = "kotoba-recipes-tokenizer-name"

    # Distributed Training
    enable_fsdp: bool = False
    low_cpu_fsdp: bool = False
    use_mpi: bool = False

    # FSDP setting
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    checkpoint_type: StateDictType = (
        StateDictType.SHARDED_STATE_DICT
    )  # alternatively can use SHARDED_STATE_DICT save one file per rank, and can resize the world-size.
    fsdp_activation_checkpointing: bool = False
    fsdp_cpu_offload: bool = False

    # training
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    clip_grad_norm: float = 1.0
    weight_decay: float = 0.1
    gamma: float = 0.85
    seed: int = 42

    num_epochs: int = 1
    use_fast_kernels: bool = False
    run_validation: bool = False
    output_dir: str = ""

    # optimizer, LR
    optimizer: str = "AdamW"
    adamw_eps: float = 1e-5
    adamw_betas: tuple[float, float] = (0.9, 0.95)
    lr: float = 1e-4
    lr_min: float = 1e-5
    lr_decay: float = 0.80  # ratio of decay
    lr_warmup: float = 0.002  # ratio of warmup
    lr_decay_style: str = "cosine"

    # Sequence Length
    use_sequence_length_schedule: bool = False
    sequence_length: int = 4096
    sequence_length_warmup_min: int = 8
    sequence_length_warmup: float = 0.15

    # precision
    use_fp16: bool = False
    use_bf16: bool = False
    mixed_precision: bool = True
    param_dtype: Optional[torch.dtype] = None

    # dataset
    dataset: str = ""
    num_workers_dataloader: int = 1

    # PEFT
    peft_method: Optional[str] = None  # None , llama_adapter, prefix
    use_peft: bool = False
    quantization: bool = False

    # freeze
    freeze_layers: bool = False
    num_freeze_layers: int = 1

    # checkpoint
    one_gpu: bool = False
    save_model: bool = True
    save_checkpoint_path: str = ""
    save_optimizer: bool = False  # will be used if using FSDP
    load_checkpoint_path: str = ""
    save_interval_iteration: int = 100

    # wandb
    wandb_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
