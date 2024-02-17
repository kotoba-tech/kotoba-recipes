#!/bin/bash

set -e

start=672
end=672
increment=5000

upload_base_dir=/model/fujii/hf_checkpoints/swallow-7b/imitative-open-asst2

for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)

  python tools/model-upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name tokyotech-llm/Swallow-7b-VE-instruct-imitative-open-asst2-2epoch-iter$(printf "%07d" $i)
done
