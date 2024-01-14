import json
from sklearn.model_selection import train_test_split  # type: ignore


def split_jsonl_dataset(file_path: str, train_ratio: float = 0.98) -> tuple[int, int]:
    # データセットを読み込む
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # データセットをランダムに分割
    train_data, val_data = train_test_split(data, train_size=train_ratio, random_state=42)
    save_dataset_dir: str = "/groups/gaf51275/llama/datasets/instruct/llm-jp-gpt4-self-instruct/"

    # 訓練データと検証データをそれぞれ別のファイルに保存
    with open(f"{save_dataset_dir}/train_data.jsonl", "w", encoding="utf-8") as file:
        for entry in train_data:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(f"{save_dataset_dir}/val_data.jsonl", "w", encoding="utf-8") as file:
        for entry in val_data:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return len(train_data), len(val_data)


file_path: str = "/groups/gaf51275/llama/datasets/instruct/llm-jp-gpt4-self-instruct/GPT4_self-instruction_data_52K.json"
train_count, val_count = split_jsonl_dataset(file_path)
print(f"Train data count: {train_count}, Validation data count: {val_count}")
