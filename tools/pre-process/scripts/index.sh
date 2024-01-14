#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=3:00:00
#$ -j y
#$ -o outputs/index/
#$ -cwd

source .env/bin/activate

INPUT_DIR=/groups/gaf51275/llama/datasets/instruct/GPT-4-LLM/data

# convert json to jsonl
python tools/pre-process/convert_json_jsonl.py \
  --data-file-path $INPUT_DIR/alpaca_gpt4_data.json

INPUT_DIR=/groups/gaf51275/llama/datasets/instruct/GPTeacher/Instruct

python tools/pre-process/convert_json_jsonl.py \
  --data-file-path $INPUT_DIR/gpt4-instruct-dedupe-only-dataset.json

python tools/pre-process/convert_json_jsonl.py \
  --data-file-path $INPUT_DIR/gpt4-instruct-similarity-0.6-dataset.json

python tools/pre-process/convert_json_jsonl.py \
  --data-file-path $INPUT_DIR/gpt4-instruct-similarity-0.7-dataset.json

python tools/pre-process/convert_json_jsonl.py \
  --data-file-path $INPUT_DIR/gpt4-instruct-similarity-0.8-dataset.json

python tools/pre-process/convert_json_jsonl.py \
  --data-file-path $INPUT_DIR/gpt4-instruct-similarity-0.9-dataset.json

# indexing
python tools/pre-process/index_dataset.py \
  --data-file-path $INPUT_DIR/gpt4-instruct-dedupe-only-dataset.jsonl

python tools/pre-process/index_dataset.py \
  --data-file-path $INPUT_DIR/gpt4-instruct-similarity-0.6-dataset.jsonl

python tools/pre-process/index_dataset.py \
  --data-file-path $INPUT_DIR/gpt4-instruct-similarity-0.7-dataset.jsonl

python tools/pre-process/index_dataset.py \
  --data-file-path $INPUT_DIR/gpt4-instruct-similarity-0.8-dataset.jsonl

python tools/pre-process/index_dataset.py \
  --data-file-path $INPUT_DIR/gpt4-instruct-similarity-0.9-dataset.jsonl

INPUT_DIR=/groups/gaf51275/llama/datasets/instruct/GPT-4-LLM/data

python tools/pre-process/index_dataset.py \
  --data-file-path $INPUT_DIR/alpaca_gpt4_data.jsonl
