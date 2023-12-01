#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=3:00:00
#$ -j y
#$ -o outputs/index/
#$ -cwd

source .env/bin/activate

python tools/pre-process/index_dataset.py \
  --data-file-path /path/to/datasets/llama2-llm-jp-corpus/v1.0.2/sample/ja_wiki/merged_train_0.jsonl
