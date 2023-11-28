#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=3:00:00
#$ -j y
#$ -o outputs/index
#$ -cwd

source .env/bin/activate

python tools/pre-process/index_dataset.py \
  --data-file-path /groups/gaf51275/llama/datasets/JParaCrawl3.0/default_instruction_following_format.jsonl

python tools/pre-process/index_dataset.py \
  --data-file-path /groups/gaf51275/llama/datasets/JParaCrawl3.0/highquality_instruction_following_format.jsonl
