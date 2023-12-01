#!/bin/bash

set -e

start=0
end=5000
increment=500

tokenizer_dir=
upload_base_dir=/bb/gaf51217/fujii/hf_checkpoints/llama-2-7b-base

for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)
  cp -r $tokenizer_dir/tokenizer* $upload_dir

  python tools/model-upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name kotoba-tech/Llama2-7b-base-iter$(printf "%07d" $i)
done
