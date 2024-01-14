#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o outputs/tokenize/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# swich virtual env
cd /bb/llm/gaf51275/llama/llama-recipes/
source .env/bin/activate

# hugginface setting cache
export HF_HOME=/bb/llm/gaf51275/.cache/huggingface

cd scripts/abci
# tokenize

for ((i=1; i<=1; i++)); do
  python llm_jp_tokenize.py $i
done
