#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=10:00:00
#$ -j y
#$ -o outputs/convert/fsdp/
#$ -cwd
# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

set -e

# swich virtual env
source .env/bin/activate

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

start=10000
end=12000
increment=1000

for ((i = start; i <= end; i += increment)); do
  ITERATION=$i
  FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)

  CHECK_POINT_PATH=/bb/gaf51217/fujii/checkpoints/llama-2-13b-chat-filtered-gbs_16-a100_2/${FORMATTED_ITERATION}
  OUTPUT_PATH=/bb/gaf51217/fujii/hf_checkpoints/kotoba/llama-2-13b-chat-filtered-gbs_16/${FORMATTED_ITERATION}

  echo "convert FSDP ${CHECK_POINT_PATH} to ${OUTPUT_PATH}"

  mkdir -p $OUTPUT_PATH

  BASE_MODEL_CHECKPOINT=/bb/llm/gaf51275/llama/huggingface-checkpoint/Llama-2-13b-hf

  mpirun -np 8 \
    --npernode 8 \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=$MASTER_PORT \
    -bind-to none -map-by slot \
    -x PATH \
    python tools/checkpoint-convert/convert_fsdp.py \
    --hf-base-model-path $BASE_MODEL_CHECKPOINT \
    --fsdp-checkpoint-path $CHECK_POINT_PATH \
    --checkpoint-output-path $OUTPUT_PATH
done
