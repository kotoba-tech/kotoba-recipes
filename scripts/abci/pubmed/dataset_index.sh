#!/bin/bash
#$ -l rt_F=1
#$ -l h_rt=0:10:00:00
#$ -j y
#$ -o outputs/dataset/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# swich virtual env
source .env/bin/activate

# train index
python tools/pre-process/index_dataset.py \
  --data-file-path /groups/gaf51275/llama/datasets/instruct/llm-jp-gpt4-self-instruct/train_data.jsonl

# valid index
python tools/pre-process/index_dataset.py \
  --data-file-path /groups/gaf51275/llama/datasets/instruct/llm-jp-gpt4-self-instruct/val_data.jsonl
