#!/bin/bash

INPUT_DIR=/model/fujii/datasets/instruction/swallow/instruction
OUTPUT_DIR=/model/fujii/datasets/instruction/swallow/instruction/only_oasst_21k

mkdir -p $OUTPUT_DIR

cat $INPUT_DIR/oasst1-21k-en.jsonl $INPUT_DIR/oasst1-21k-ja.jsonl > $OUTPUT_DIR/merged.jsonl

echo "Merged dataset is saved at $OUTPUT_DIR/merged.jsonl"

# swich virtual env
source .env/bin/activate

python scripts/mdx/dataset/shuffle_and_split.py \
  --input $OUTPUT_DIR/merged.jsonl \
  --output $OUTPUT_DIR

# shuffling len(jsonl_data)=42328
# shuffled len(jsonl_data)=42328
# len(train_data)=40211, len(val_data)=2117
