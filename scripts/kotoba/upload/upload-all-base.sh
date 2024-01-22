#!/bin/bash

set -e

start=20000
end=25000
increment=5000

upload_base_dir=/bb/llm/gaf51275/llama/converted-hf-checkpoint/mistral-7B-VE/okazaki-cc

# for ループで指定された範囲と増分を使用
for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)

  python scripts/kotoba/upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name tokyotech-llm/Mistral-7B-VE-okazaki-cc-iter$(printf "%07d" $i)
done
