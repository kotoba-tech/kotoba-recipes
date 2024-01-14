import pandas as pd

df1 = pd.read_json(
    '/groups/gaf51275/llama/datasets/instruct/databricks-dolly-15k-ja/databricks-dolly-15k-ja.json',
)
print(f'databricks-dolly-15k-ja Rows: {len(df1)}')

df2 = pd.read_json(
    '/groups/gaf51275/llama/datasets/instruct/refined_hh-rlhf-49k-ja/mpt_hhrlhf_49k_ja.jsonl', lines=True
)
print(f'refined_hh-rlhf-49k-ja Rows: {len(df2)}')

df3 = pd.read_json(
    '/groups/gaf51275/llama/datasets/instruct/refined_oasst1-89k-ja/oasst1_ja_converted.json'
)
print(f"refined_oasst1-89k-ja Rows: {len(df3)}")

combined_df = pd.concat([df1, df2, df3])
shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)

shuffled_df.to_json('/groups/gaf51275/llama/datasets/instruct/abci_llama_instruct_dataset/shuffled_combined.jsonl', orient='records', lines=True, force_ascii=False)
print(f"Combined Rows: {len(shuffled_df)}")
