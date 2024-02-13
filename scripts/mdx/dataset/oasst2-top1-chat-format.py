import argparse
import os
from tqdm import tqdm
import json


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = arg_parse()

    jsonl_data: list = []
    with open(args.input, "r") as f:
        for line in f:
            jsonl_data.append(json.loads(line))

    for conversations in tqdm(jsonl_data):
        for key, value in conversations.items():
            print(f"{key}={value}")
        break


if __name__ == "__main__":
    main()
