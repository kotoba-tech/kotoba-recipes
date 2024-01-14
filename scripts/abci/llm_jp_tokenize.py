from typing import List
import datasets
import sys

from transformers import LlamaTokenizer
from tqdm.contrib.concurrent import process_map

from itertools import chain


class Concatenator(object):
    def __init__(self, chunk_size=2048):
        self.chunk_size: int = chunk_size
        self.residual = {"input_ids": [], "attention_mask": []}

    def __call__(self, batch):
        concatenated_samples = {k: v + list(chain(*batch[k])) for k, v in self.residual.items()}

        total_length = len(concatenated_samples[list(concatenated_samples.keys())[0]])

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [v[i : i + self.chunk_size] for i in range(0, chunk_num * self.chunk_size, self.chunk_size)]
                for k, v in concatenated_samples.items()
            }
            self.residual = {k: v[(chunk_num * self.chunk_size) :] for k, v in concatenated_samples.items()}
        else:
            result = concatenated_samples
            self.residual = {k: [] for k in concatenated_samples.keys()}

        result["labels"] = result["input_ids"].copy()

        return result


def load_dataset(split: str, tokenizer, return_dict, paths: List[str]) -> None:
    raw_dataset: datasets.DatasetDict = datasets.load_dataset(  # type: ignore
        path="json",
        data_files=paths,
        num_proc=2,
    )

    def tokenize_map(sample):
        return tokenizer(sample["text"])

    dataset = (
        raw_dataset["train"]
        .map(
            process_map(tokenize_map, raw_dataset["train"], max_workers=4),
            batched=True,
            remove_columns=list(raw_dataset["train"].features),
        )
        .map(Concatenator(chunk_size=4096), batched=True)
    )
    return_dict[split] = dataset[split]


def get_llm_jp_dataset(tokenizer, split: str = "train", index: int = 0):
    if split == "train":
        train_path: str = (
            f"/bb/llm/gaf51275/llama/datasets/llama2-llm-jp-corpus/v1.0.2/sample/ja_cc/merged_train_{index}.jsonl"
        )

        dataset_paths: list[str] = [train_path]
        print(f"processing {train_path}")

        raw_dataset: datasets.DatasetDict = datasets.load_dataset(  # type: ignore
            path="json",
            data_files=dataset_paths,
            num_proc=64,
        )
        dataset = (
            raw_dataset["train"]
            .map(
                lambda sample: tokenizer(sample["text"]),
                batched=True,
                remove_columns=list(raw_dataset["train"].features),
                num_proc=64,
            )
            .map(Concatenator(chunk_size=4096), batched=True, num_proc=64)
        )
        return dataset
    else:
        dataset_paths: list[str] = [
            "/bb/llm/gaf51275/llama/datasets/llama2-llm-jp-corpus/v1.0.2/sample/ja_cc/merged_val_0.jsonl",
            "/bb/llm/gaf51275/llama/datasets/llama2-llm-jp-corpus/v1.0.2/sample/ja_wiki/merged_val_0.jsonl",
        ]
        raw_dataset: datasets.DatasetDict = datasets.load_dataset(  # type: ignore
            path="json",
            data_files=dataset_paths,
            num_proc=64,
        )
        dataset = (
            raw_dataset["train"]
            .map(
                lambda sample: tokenizer(sample["text"]),
                batched=True,
                remove_columns=list(raw_dataset["train"].features),
                num_proc=64,
            )
            .map(Concatenator(chunk_size=4096), batched=True, num_proc=64)
        )
        return dataset


def main() -> None:
    args: list[str] = sys.argv
    print(f"index: {args[1]}")

    tokenizer = LlamaTokenizer.from_pretrained(
        "/bb/llm/gaf51275/jalm/jalm-tokenizer-private/tokenizer/jalm_llama_clueweb/merged_tokenizer_hf"
    )
    tokenizer.add_special_tokens(
        {
            "pad_token": "<PAD>",
        }
    )

    tokenized_dataset = get_llm_jp_dataset(tokenizer, split="train", index=args[1])  # type: ignore
    tokenized_dataset.save_to_disk(f"/bb/llm/gaf51275/llama/tokenized/tokenized_ja_cc_{0}.arrow")


if __name__ == "__main__":
    main()
