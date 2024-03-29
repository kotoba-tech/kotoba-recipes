import copy
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from pathlib import Path
from llama_recipes.utils.distributed import print_rank_0

from megatron_lm.megatron.global_vars import get_args, set_sampler


PROMPT_DICT = {
    "prompt_input": (
        "以下に、あるタスクを説明する指示があり、それに付随する入力が更なる文脈を提供しています。"
        "リクエストを適切に完了するための回答を記述してください。\n\n"
        "### 指示:\n{instruction}\n\n### 入力:\n{input}\n\n### 応答:"
    ),
    "prompt_no_input": (
        "以下に、あるタスクを説明する指示があります。" "リクエストを適切に完了するための回答を記述してください。\n\n" "### 指示:\n{instruction}\n\n### 応答:"
    ),
}


class InstructDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
    ) -> None:
        args = get_args()

        self.data_path: str = data_path
        self.max_words: int = args.seq_length
        self.tokenizer = tokenizer

        # index file
        dataset_dir = Path(self.data_path).parent
        index_cache_dir = dataset_dir / ".index_cache"
        os.makedirs(index_cache_dir, exist_ok=True)
        index_file_path = index_cache_dir / str(os.path.basename(self.data_path)).replace(".jsonl", ".idx")
        self.index_file_path: str = str(index_file_path)

        try:
            with open(self.index_file_path, "r", encoding="utf-8") as f:
                self.indexes: list[int] = [int(line.strip()) for line in f]
        except Exception as e:
            print(f"index file error: {e}")
            exit(1)

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        with open(self.data_path, "r", encoding="utf-8") as file:
            offset: int = self.indexes[index]
            file.seek(offset)
            try:
                line = file.readline()
            except Exception as e:
                print(f"index={index}, offset={offset}, error={e}")
                exit(1)

            try:
                ann: dict[str, str] = json.loads(line)
            except Exception as e:
                print(f"index={index}, offset={offset}, line={line}, error={e}")
                exit(1)

        if ann.get("input", "") == "":
            prompt: str = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)

        example: str = prompt + ann["output"]
        encoded_prompt: torch.Tensor = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
        encoded_example: list[int] = self.tokenizer.encode(example)
        encoded_example.append(self.tokenizer.eos_token_id)  # type: ignore
        encoded_tensor_example: torch.Tensor = torch.tensor(encoded_example, dtype=torch.int64)

        padding: int = self.max_words - encoded_tensor_example.shape[0]
        if padding > 0:
            encoded_tensor_example = torch.cat((encoded_tensor_example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            encoded_tensor_example = encoded_tensor_example[: self.max_words]

        labels = copy.deepcopy(encoded_tensor_example)
        # promptの長さ分だけ -1 で埋める -> 損失関数で無視するようになる
        labels[: len(encoded_prompt)] = -1
        # 0より大きい(ge)かどうかの真偽値でmaskを作成
        example_mask = encoded_tensor_example.ge(0)
        label_mask = labels.ge(0)

        if torch.all(label_mask == 0):
            random_index: int = np.random.randint(0, len(self.indexes))
            self.__getitem__(random_index)

        # ~example_mask -> paddingの部分を 0 で埋める
        encoded_tensor_example[~example_mask] = 0
        # ~label_mask -> prompt の部分を ignore_index で埋める
        labels[~label_mask] = IGNORE_INDEX

        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": encoded_tensor_example,
            "labels": labels,
            "attention_mask": example_mask,
        }


def worker_init_fn(worker_id: int) -> None:
    import random

    args = get_args()

    worker_seed = args.seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_instruction_tuning_dataloader(
    tokenizer: PreTrainedTokenizer,
    data_path: str,
    train: bool = False,
) -> torch.utils.data.DataLoader:
    from llama_recipes.utils.sequence_length_warmup import CustomDistributedSampler
    from llama_recipes.utils.checkpoint import load_sampler_state_dict

    args = get_args()

    instruction_dataset = InstructDataset(
        tokenizer=tokenizer,
        data_path=data_path,
    )

    if train:
        args.instruction_dataset_size = len(instruction_dataset)
        print_rank_0(f"Instruction dataset size: {args.instruction_dataset_size}")

    train_sampler = CustomDistributedSampler(
        dataset=instruction_dataset,
        rank=torch.distributed.get_rank(),
        num_replicas=torch.distributed.get_world_size(),
        shuffle=True,
        seed=args.seed,
    )

    if args.load:
        load_sampler_state_dict(sampler=train_sampler, path=args.load)

    set_sampler(sampler=train_sampler)

    return torch.utils.data.DataLoader(
        instruction_dataset,
        batch_size=args.micro_batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
