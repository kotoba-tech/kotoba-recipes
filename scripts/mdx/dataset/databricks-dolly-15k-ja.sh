#!/bin/bash

# swich virtual env
source .env/bin/activate

DATASET_DIR=/model/fujii/datasets/instruction/swallow

python scripts/mdx/dataset/databricks-dolly-15k-ja.py \
  --input ${DATASET_DIR}/databricks-dolly-15k-ja/databricks-dolly-15k-ja.jsonl \
  --output /model/fujii/datasets/instruction/swallow/instruction/databricks-dolly-15k-ja.jsonl
