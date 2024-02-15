#!/bin/bash

INPUT_DIR=/model/fujii/datasets/instruction/swallow/instruction
OUTPUT_DIR=/model/fujii/datasets/instruction/swallow/instruction/hh-rlhf

mkdir -p $OUTPUT_DIR

cat $INPUT_DIR/databricks-dolly-15k-en.jsonl $INPUT_DIR/databricks-dolly-15k-ja.jsonl $INPUT_DIR/oasst1-21k-en.jsonl $INPUT_DIR/oasst1-21k-ja.jsonl $INPUT_DIR/hh-rlhf-12k-ja.jsonl > $OUTPUT_DIR/merged.jsonl

echo "Merged dataset is saved at $OUTPUT_DIR/merged.jsonl"

# swich virtual env
source .env/bin/activate

python scripts/mdx/dataset/shuffle_and_split.py \
  --input $OUTPUT_DIR/merged.jsonl \
  --output $OUTPUT_DIR

# shuffling len(jsonl_data)=84350
# shuffled len(jsonl_data)=84350
# len(train_data)=80132, len(val_data)=4218
