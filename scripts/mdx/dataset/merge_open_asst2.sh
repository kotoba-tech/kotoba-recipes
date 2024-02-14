#!/bin/bash

INPUT_DIR=/model/fujii/datasets/instruction/swallow/instruction
OUTPUT_DIR=/model/fujii/datasets/instruction/swallow/instruction/open_asst2

mkdir -p $OUTPUT_DIR

cat $INPUT_DIR/databricks-dolly-15k-en.jsonl $INPUT_DIR/databricks-dolly-15k-ja.jsonl $INPUT_DIR/oasst1-21k-en.jsonl $INPUT_DIR/oasst1-21k-ja.jsonl /model/fujii/datasets/instruction/swallow/oasst2-top1-en/oasst2-top1-en_instruction_tuning_format.jsonl > $OUTPUT_DIR/merged.jsonl

echo "Merged dataset is saved at $OUTPUT_DIR/merged.jsonl"

# swich virtual env
source .env/bin/activate

python scripts/mdx/dataset/shuffle_and_split.py \
  --input $OUTPUT_DIR/merged.jsonl \
  --output $OUTPUT_DIR

# shuffling len(jsonl_data)=80941
# shuffled len(jsonl_data)=80941
# len(train_data)=76893, len(val_data)=4048
