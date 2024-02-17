#!/bin/bash

# swich virtual env
source .env/bin/activate

TARGET_DIR=/model/fujii/datasets/instruction/swallow/instruction/open_asst2

python tools/pre-process/index_dataset.py \
  --data-file-path $TARGET_DIR/train.jsonl

python tools/pre-process/index_dataset.py \
  --data-file-path $TARGET_DIR/val.jsonl