import datasets


dataset = datasets.load_dataset(
    path="/groups/gaf51275/llama/datasets/instruct/hh-rlhf-49k-ja",
)
dataset.set_format(type="pandas")  # type: ignore
df = dataset["train"][:]  # type: ignore

print(f"Original size: {len(df)}")
df = df[df["ng_translation"] != "1"].drop(["ng_translation", "index"], axis=1).reset_index()  # type: ignore
print(f"Filtered size: {len(df)}")

df.to_json(
    "/groups/gaf51275/llama/datasets/instruct/refined_hh-rlhf-49k-ja/mpt_hhrlhf_49k_ja.jsonl",
    orient="records",
    lines=True,
)
