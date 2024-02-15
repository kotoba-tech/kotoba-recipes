#!/bin/bash

set -e

# swich virtual env
source .env/bin/activate

# fsdp
start=2234
end=2234
increment=5000

for ((i = start; i <= end; i += increment)); do
  ITERATION=$i
  FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)

  CHECK_POINT_PATH=/model/fujii/checkpoints/Mistral-7b-VE-algebric-stack/ichikara-lr_1e-5-minlr_1e-6/${FORMATTED_ITERATION}/model.pt
  OUTPUT_PATH=/model/fujii/hf_checkpoints/mistral-7b-ve-algebric-stack/ichikara/${FORMATTED_ITERATION}

  echo "convert ${CHECK_POINT_PATH} to ${OUTPUT_PATH}"

  mkdir -p $OUTPUT_PATH

  BASE_MODEL_CHECKPOINT=/model/fujii/hf_checkpoints/Mistral-7B-VE-algebraic-stack-lr_2e-5-minlr_6.6e-7_warmup_1000-iter0025000/

  python tools/checkpoint-convert/convert_ckpt.py \
    --model $BASE_MODEL_CHECKPOINT \
    --ckpt $CHECK_POINT_PATH \
    --out $OUTPUT_PATH \
    --sequence-length 4096
done
