#!/bin/bash

# swich virtual env
source .env/bin/activate

TARGET_DIR=/model/fujii/datasets/instruction/swallow/instruction/only_oasst_21k

python tools/pre-process/index_dataset.py \
  --data-file-path $TARGET_DIR/train.jsonl

python tools/pre-process/index_dataset.py \
  --data-file-path $TARGET_DIR/val.jsonl
