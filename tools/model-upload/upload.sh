#!/bin/bash

set -e

start=2234
end=2234
increment=5000

upload_base_dir=/model/fujii/hf_checkpoints/swallow-13b/ichikara

for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)

  python tools/model-upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name tokyotech-llm/Swallow-13b-VE-instruct-ichikara-2epoch-iter$(printf "%07d" $i)
done
