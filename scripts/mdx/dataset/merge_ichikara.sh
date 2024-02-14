#!/bin/bash

INPUT_DIR=/model/fujii/datasets/instruction/swallow/instruction
OUTPUT_DIR=/model/fujii/datasets/instruction/swallow/instruction/merge_ichikara

mkdir -p $OUTPUT_DIR

cat $INPUT_DIR/databricks-dolly-15k-en.jsonl $INPUT_DIR/databricks-dolly-15k-ja.jsonl $INPUT_DIR/oasst1-21k-en.jsonl $INPUT_DIR/oasst1-21k-ja.jsonl $INPUT_DIR/ichikara-1-ja.jsonl $INPUT_DIR/ichikara-2-ja.jsonl $INPUT_DIR/ichikara-3-ja.jsonl $INPUT_DIR/ichikara-4-ja.jsonl $INPUT_DIR/ichikara-5-ja.jsonl > $OUTPUT_DIR/merged.jsonl

echo "Merged dataset is saved at $OUTPUT_DIR/merged.jsonl"

# swich virtual env
source .env/bin/activate

python scripts/mdx/dataset/shuffle_and_split.py \
  --input $OUTPUT_DIR/merged.jsonl \
  --output $OUTPUT_DIR

# Merged dataset is saved at /model/fujii/datasets/instruction/swallow/instruction/merge_ichikara/merged.jsonl
# shuffling len(jsonl_data)=75253
# shuffled len(jsonl_data)=75253
# len(train_data)=71490, len(val_data)=3763
