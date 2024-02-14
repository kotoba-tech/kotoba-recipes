#!/bin/bash

# swich virtual env
source .env/bin/activate

# distributed settings
JOB_ID=$(date +%s%N)
export MASTER_ADDR=10.130.184.10
export MASTER_PORT=12806

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=8
NODE_TYPE="a100"

NUM_NODES=16
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

SGE_JOB_HOSTLIST=scripts/mdx/hostfile_16
export SGE_JOB_HOSTLIST

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r line; do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done <"$SGE_JOB_HOSTLIST" >"$HOSTFILE_NAME"

# training config
SEQ_LENGTH=4096
DATA_PARALLEL_SIZE=$NUM_GPUS

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=64

# optimizer config
LR=1e-5
MIN_LR=1e-6
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# checkpoint & tokenizer
TOKENIZER_MODEL=/model/fujii/hf_checkpoints/Swallow-70b-hf/tokenizer.model
CHECKPOINT_DIR=/model/fujii/hf_checkpoints/Swallow-70b-hf/
CHECKPOINT_SAVE_DIR="/model/fujii/checkpoints/Swallow-70b/lr_${LR}-minlr_${MIN_LR}"

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config

DATASET_DIR=/model/fujii/datasets/instruction/swallow/instruction/baseline

TRAIN_DATA_PATH=${DATASET_DIR}/train.jsonl
VALID_DATA_PATH=${DATASET_DIR}/val.jsonl

# job name
JOB_NAME="Swallow-70b-VE-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}"

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none -map-by slot \
  -x PATH \
  python examples/finetuning.py \
  --seq-length ${SEQ_LENGTH} \
  --sliding-window-size ${SEQ_LENGTH} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --hf-transformer-model-dir ${CHECKPOINT_DIR} \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --instruction-train-data-path ${TRAIN_DATA_PATH} \
  --instruction-valid-data-path ${VALID_DATA_PATH} \
  --epoch 2 \
  --train-iters 500000 \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style cosine \
  --weight-decay ${WEIGHT_DECAY} \
  --grad-clip-norm ${GRAD_CLIP} \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --adam-eps 1e-6 \
  --save-interval 500 \
  --eval-interval 100 \
  --eval-iters 10 \
  --bf16 \
  --mixed-precision \
  --base-model ${CHECKPOINT_DIR} \
  --save ${CHECKPOINT_SAVE_DIR} \
  --load ${CHECKPOINT_SAVE_DIR} \
  --low-cpu-fsdp \
  --fsdp-cpu-offload \
  --sharding-strategy FULL_SHARD \
  --checkpoint-type LOCAL_STATE_DICT \
  --fsdp-activation-checkpointing \
  --instruction-tuning \
  --save-sampler-state \
  --use-mpi \
  --wandb-entity "prj-jalm" \
  --wandb-project "Llama-2-70b-instruct" \
  --wandb-name "${JOB_NAME}"
