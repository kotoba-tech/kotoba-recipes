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
        instruction_text: str = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"

        for index, conversation in enumerate(conversations["conversations"]):

            if conversation["from"] == "human":
                instruction_text += "### Instruction:\n" + conversation["value"] + "\n\n"
            elif conversation["from"] == "gpt":
                if len(conversations["conversations"]) == index + 1:
                    instruction_text += "### Response:\n"
                    instruction_data.append(instruction_text)
                    output_data.append(conversation["value"])
                else:
                    instruction_text += "### Response:\n" + conversation["value"] + "\n\n"
            else:
                print(f"invalid conversation={conversation}")

        if args.debug:
            print(f"instruction_data={instruction_data[-1]}output_data={output_data[-1]}")
            exit(0)

    print(f"\n\nlen(instruction_data)={len(instruction_data)}, len(output_data)={len(output_data)}\n\n")

    # save
    with open(args.output, "w") as f:
        for instruction, output in zip(instruction_data, output_data):
            f.write(json.dumps({"instruction": instruction, "output": output}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
