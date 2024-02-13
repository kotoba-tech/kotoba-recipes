#!/bin/bash

# swich virtual env
source .env/bin/activate

DATASET_DIR=/model/fujii/datasets/instruction/swallow

python scripts/mdx/dataset/databricks-dolly-15k-en.py \
  --input ${DATASET_DIR}/databricks-dolly-15k/databricks-dolly-15k.jsonl \
  --output /model/fujii/datasets/instruction/swallow/instruction/databricks-dolly-15k-en.jsonl
