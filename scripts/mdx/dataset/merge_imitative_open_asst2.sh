#!/bin/bash

INPUT_DIR=/model/fujii/datasets/instruction/swallow/instruction
OUTPUT_DIR=/model/fujii/datasets/instruction/swallow/instruction/imitative-open-asst2

mkdir -p $OUTPUT_DIR

cat $INPUT_DIR/oasst1-imitative-ja.jsonl /model/fujii/datasets/instruction/swallow/oasst2-top1-en/oasst2-top1-en_instruction_tuning_format.jsonl > $OUTPUT_DIR/merged.jsonl

echo "Merged dataset is saved at $OUTPUT_DIR/merged.jsonl"

# swich virtual env
source .env/bin/activate

python scripts/mdx/dataset/shuffle_and_split.py \
  --input $OUTPUT_DIR/merged.jsonl \
  --output $OUTPUT_DIR

# shuffling len(jsonl_data)=22699
# shuffled len(jsonl_data)=22699
# len(train_data)=21564, len(val_data)=1135
