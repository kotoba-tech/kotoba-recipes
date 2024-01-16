import os
import time
from pkg_resources import packaging  # type: ignore
from contextlib import nullcontext

import torch
import torch.cuda.nccl as nccl
from torch import distributed as torch_distributed  # noqa: F401
from torch.utils.data import DataLoader
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.nn.utils import clip_grad_norm_  # type: ignore
from tqdm import tqdm
from llama_recipes.policies import fpSixteen, bfSixteen, bfSixteen_mixed, get_decoder_layer_wrapper
from llama_recipes.utils.memory_utils import MemoryTrace
from llama_recipes.utils.wandb_utils import log_model_info, log_wandb
from llama_recipes.utils.checkpoint import save_checkpoint, get_latest_iteration

from typing import Optional, Any
import wandb
from megatron_lm.megatron.global_vars import get_args, get_tokenizer


def train(
    model,
    train_dataloader,
    eval_dataloader,
    optimizer: torch.optim.AdamW,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    gradient_accumulation_steps: int,
    local_rank: Optional[int] = None,
    rank: Optional[int] = None,
) -> None:
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps:
            The number of steps to accumulate gradients before performing
            a backward/update operation
        local_rank: The rank of the current node in a distributed setting
        rank:

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    args = get_args()
    # Create a gradient scaler for fp16
    if args.fp16:
        scaler = ShardedGradScaler()

    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = local_rank if local_rank is not None else 0
    autocast = torch.cuda.amp.autocast if args.fp16 else nullcontext  # type: ignore

    train_prep: list[torch.Tensor] = []  # train perplexity
    train_loss: list[float] = []  # train loss
    val_prep: list[torch.Tensor] = []  # validation perplexity
    val_loss: list[float] = []  # validation loss
    epoch_times: list[float] = []
    checkpoint_times: list[float] = []
    best_val_loss = float("inf")

    # set model info
    if rank == 0 and args.wandb_name:
        log_model_info(model)

    last_epoch: int = 0
    last_iteration: int = 0
    total_iterations: int = len(train_dataloader) // gradient_accumulation_steps

    # load checkpoint(model, optimizer, scheduler, sampler)
    if args.load != "":
        last_iteration = get_latest_iteration(args.load)
        last_epoch = last_iteration // total_iterations

    wandb_iteration: int = 0
    for epoch in range(last_epoch, num_epochs):
        epoch_start_time = time.perf_counter()
        iteration_start_time = time.perf_counter()

        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss: float = 0.0
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch}",
                total=total_iterations,
                disable=(rank != 0),
            )

            accumulation_loss: float = 0.0

            for step, batch in enumerate(train_dataloader_iter, start=next_step):
                model.train()
                wandb_iteration = epoch * total_iterations + step // gradient_accumulation_steps

                for key in batch.keys():
                    batch[key] = batch[key].to(local_rank)

                with autocast():
                    loss: torch.Tensor = model(**batch).loss
                loss = loss / gradient_accumulation_steps

                if args.fp16:
                    # if fp16 is enabled, use gradient scaler to handle gradient update
                    scaler.scale(loss).backward()  # type: ignore

                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        scaler.step(optimizer)  # type: ignore (suppress ubound error)
                        scaler.update()  # type: ignore (suppress ubound error)
                        optimizer.zero_grad()
                        lr_scheduler.step()
                        pbar.update(step // gradient_accumulation_steps)
                else:
                    # regular back propagation when fp16 is not used
                    loss.backward()

                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_scheduler.step()
                        pbar.update(step // gradient_accumulation_steps)

                total_loss += loss.item()
                accumulation_loss += loss.item()

                # gradient clipping
                if args.grad_clip_norm > 0:
                    clip_grad_norm_(model.parameters(), args.grad_clip_norm)

                pbar.set_description(
                    f"Training Epoch: {epoch}/{num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.item() * gradient_accumulation_steps}, lr: {optimizer.param_groups[0]['lr']:.6f}, accumulation_step: {step % gradient_accumulation_steps + 1}/{gradient_accumulation_steps}, iteration: {wandb_iteration})"  # noqa: E501
                )

                if args.wandb_name and (step + 1) % gradient_accumulation_steps == 0:
                    avg_loss = torch.tensor(accumulation_loss).to(local_rank)  # type: ignore
                    torch_distributed.all_reduce(tensor=avg_loss, op=torch_distributed.ReduceOp.SUM)
                    avg_loss = avg_loss / world_size

                    if rank == 0:
                        log_wandb(
                            batch=batch,
                            model=model,
                            accumulation_loss=avg_loss,
                            optimizer=optimizer,
                            epoch=epoch,
                            step=step,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            world_size=world_size,
                            iteration_start_time=iteration_start_time,
                            wandb_iteration=wandb_iteration,
                        )
                    accumulation_loss = 0.0
                    iteration_start_time = time.perf_counter()

                if (wandb_iteration + 1) % args.eval_interval == 0:
                    # validation
                    eval_ppl, eval_loss = evaluation(
                        model=model,
                        eval_dataloader=eval_dataloader,  # type: ignore
                        local_rank=local_rank,
                        wandb_log=True,
                    )
                    if rank == 0:
                        wandb.log(
                            {"evaluation/val_loss": eval_loss, "evaluation/val_ppl": eval_ppl},
                            step=wandb_iteration + 1,
                        )
                if (wandb_iteration + 1) % args.save_interval == 0:
                    # checkpoint save
                    save_checkpoint(
                        model=model,  # type: ignore
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        path=args.save,
                        iteration=wandb_iteration + 1,
                    )

        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if torch.cuda.device_count() > 1:
            total_loss = torch.tensor(total_loss).to(local_rank)  # type: ignore
            torch_distributed.all_reduce(total_loss, op=torch_distributed.ReduceOp.SUM)
        train_epoch_loss: float = total_loss / len(train_dataloader) * gradient_accumulation_steps
        train_epoch_loss: float = train_epoch_loss / world_size
        train_perplexity: torch.Tensor = torch.exp(train_epoch_loss)  # type: ignore

        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)

        if rank == 0:
            print(f"Max CUDA memory allocated was {memtrace.peak} GB")
            print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
            print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
            print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
            print(
                f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB"  # noqa: E501
            )

        eval_ppl, eval_epoch_loss = evaluation(
            model=model,
            eval_dataloader=eval_dataloader,  # type: ignore
            local_rank=local_rank,
        )
        checkpoint_start_time = time.perf_counter()

        torch_distributed.barrier()
        save_checkpoint(
            model=model,  # type: ignore
            optimizer=optimizer,
            scheduler=lr_scheduler,
            path=args.save,
            iteration=(epoch + 1) * len(train_dataloader) // gradient_accumulation_steps,
        )
        torch_distributed.barrier()

        checkpoint_end_time = time.perf_counter() - checkpoint_start_time
        checkpoint_times.append(checkpoint_end_time)
        if eval_epoch_loss < best_val_loss:
            best_val_loss = eval_epoch_loss

            if rank == 0:
                print(f"best eval loss on epoch {epoch} is {best_val_loss}")

        val_loss.append(best_val_loss)
        val_prep.append(eval_ppl)
        if rank == 0:
            print(
                f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s"  # noqa: E501
            )


def evaluation(
    model,
    eval_dataloader: DataLoader,
    local_rank: int,
    wandb_log: bool = False,
):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting

    Returns: eval_ppl, eval_epoch_loss
    """
    world_size: int = 1  # suppress ubound error
    if not wandb_log:
        world_size = int(os.environ["WORLD_SIZE"])

    model.eval()
    eval_preds = []
    eval_loss = 0.0  # Initialize evaluation loss

    with MemoryTrace() as memtrace:  # noqa: F841
        for step, batch in enumerate(tqdm(eval_dataloader, colour="green", desc="evaluating Epoch")):
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
            # Decode predictions and add to evaluation predictions list
            preds = torch.argmax(outputs.logits, -1)
            tokenizer = get_tokenizer()
            eval_preds.extend(tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True))

    if torch.cuda.device_count() > 1 and not wandb_log:
        torch_distributed.all_reduce(eval_loss, op=torch_distributed.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss: float = eval_loss / len(eval_dataloader)
    eval_epoch_loss = eval_epoch_loss / world_size
    eval_ppl = torch.exp(eval_epoch_loss)  # type: ignore

    # Print evaluation metrics
    if local_rank == 0:
        print(f" {eval_ppl=} {eval_epoch_loss=}")

    return eval_ppl, eval_epoch_loss


def freeze_transformer_layers(model, num_layer: int) -> None:
    """transformerの一部のlayerをfreezeする

    Args:
        model: モデル
        num_layer (int): freezeするlayerの数 [0〜 num_layer)
    """
    for i, layer in enumerate(model.model.layers):
        if i < num_layer:
            for param in layer.parameters():
                param.requires_grad = False


def check_frozen_layers_peft_model(model) -> None:
    for i, layer in enumerate(model.base_model.model.model.layers):
        for name, param in layer.named_parameters():
            print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")


def setup_environ_flags(rank: int) -> None:
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    if rank == 0:
        print("--> Running with torch torch_distributed debug set to detail")


def cleanup() -> None:
    """Clean up the process group after training"""
    torch_distributed.destroy_process_group()


def clear_gpu_cache(rank: Optional[int] = None) -> None:
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print("Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def get_parameter_dtypes(model) -> dict[Any, Any]:
    """Get the data types of model parameters"""
    parameter_dtypes: dict[Any, Any] = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes


def print_model_size(model, model_name: str, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        rank (int, optional): Current process's rank. Defaults to 0.
    """

    if rank == 0:
        print(f"--> Model {model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {model_name} has {total_params / 1e6} Million params\n")


def get_policies(rank: int, model_name: str):
    """Get the policies for mixed precision and fsdp wrapping"""

    verify_bfloat_support: bool = (
        torch.version.cuda  # type: ignore
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)  # type: ignore
        and torch_distributed.is_nccl_available()
        and nccl.version() >= (2, 10)
    )

    mixed_precision_policy = None
    wrapping_policy = None

    args = get_args()

    # Mixed precision
    if args.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not args.fp16 and args.param_dtype == "fp32":
            mixed_precision_policy = bfSixteen_mixed
            if rank == 0:
                print("\nBFloat16 enabled for mixed precision - using bfSixteen_mixed policy\n", flush=True)
        elif bf16_ready and not args.fp16:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print("\nBFloat16 enabled for mixed precision - using bfSixteen policy\n", flush=True)
        elif args.fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print("\nFP16 enabled\n", flush=True)
        else:
            print("bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_decoder_layer_wrapper(model_name=model_name)
    return mixed_precision_policy, wrapping_policy
