import argparse
import os
from tqdm import tqdm
import json


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = arg_parse()

    jsonl_data: list = []
    with open(args.input, "r") as f:
        for line in f:
            jsonl_data.append(json.loads(line))

    instruction_data = []
    output_data = []
    for conversations in tqdm(jsonl_data):
        instruction_text: str = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n"

        for index, conversation in enumerate(conversations["conversations"]):

            if conversation["from"] == "human":
                instruction_text += "### 指示:\n" + conversation["value"] + "\n\n"
            elif conversation["from"] == "mixtral":
                if len(conversations["conversations"]) == index + 1:
                    instruction_text += "### 応答:\n"
                    instruction_data.append(instruction_text)
                    output_data.append(conversation["value"])
                else:
                    instruction_text += "### 応答:\n" + conversation["value"] + "\n\n"
            else:
                print(f"invalid conversation={conversation}")

        if args.debug:
            print(f"instruction_data={instruction_data[-1]}output_data={output_data[-1]}")
            if len(instruction_data) > 10:
                exit(0)

    print(f"\n\nlen(instruction_data)={len(instruction_data)}, len(output_data)={len(output_data)}\n\n")

    # save
    with open(args.output, "w") as f:
        for instruction, output in zip(instruction_data, output_data):
            f.write(json.dumps({"instruction": instruction, "output": output}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
