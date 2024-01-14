#!/bin/bash

set -e

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# swich virtual env
source .env/bin/activate

start=398
end=1990
increment=398


upload_base_dir=/groups/gaf51275/llama/checkpoints/hf_checkpoints/instruct-gpt-4/llama-2-13b-base-extended-cc

# for ループで指定された範囲と増分を使用
for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)

  python scripts/kotoba/upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name tokyotech-llm/instruct-gpt-4-llama-2-13b-base-extended-cc-iter$(printf "%07d" $i)
done
