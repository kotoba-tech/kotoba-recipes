#!/bin/bash

# swich virtual env
source .env/bin/activate

DATASET_DIR=/model/fujii/datasets/instruction/swallow

python scripts/mdx/dataset/ichikara.py \
  --input ${DATASET_DIR}/ichikara/ichikara-instruction-003-001-5.1.json \
  --output /model/fujii/datasets/instruction/swallow/instruction/ichikara-1-ja.jsonl

python scripts/mdx/dataset/ichikara.py \
  --input ${DATASET_DIR}/ichikara/ichikara-instruction-003-001-5.2.json \
  --output /model/fujii/datasets/instruction/swallow/instruction/ichikara-2-ja.jsonl

python scripts/mdx/dataset/ichikara.py \
  --input ${DATASET_DIR}/ichikara/ichikara-instruction-003-001-1.json \
  --output /model/fujii/datasets/instruction/swallow/instruction/ichikara-3-ja.jsonl

python scripts/mdx/dataset/ichikara.py \
  --input ${DATASET_DIR}/ichikara/ichikara-instruction-003-001-2.1.json \
  --output /model/fujii/datasets/instruction/swallow/instruction/ichikara-4-ja.jsonl

python scripts/mdx/dataset/ichikara.py \
  --input ${DATASET_DIR}/ichikara/ichikara-instruction-003-001-2.2.json \
  --output /model/fujii/datasets/instruction/swallow/instruction/ichikara-5-ja.jsonl
