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

start=917
end=4585
increment=917

tokenizer_dir=/bb/llm/gaf51275/llama/from_megatron_hf_checkpoints/hf_checkpoints/Llama2-13b-base-extended-cc/iter_0025000
upload_base_dir=/groups/gaf51275/llama/checkpoints/hf_checkpoints/instruct/llama-2-13b-base-extended-cc

# for ループで指定された範囲と増分を使用
for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)
  cp -r $tokenizer_dir/tokenizer* $upload_dir

  python scripts/kotoba/upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name tokyotech-llm/instruct-abci-llama-2-13b-base-extended-cc-iter$(printf "%07d" $i)
done
