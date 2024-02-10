#!/bin/bash
#YBATCH -r rtx6000-ada_4
#SBATCH --job-name=instruction
#SBATCH --time=2:00:00
#SBATCH --output outputs/instrction/%j.out
#SBATCH --error errors/instruction/%j.err
. /etc/profile.d/modules.sh
module load cuda/11.8
module load cudnn/cuda-11.x/8.9.0
module load nccl/cuda-11.7/2.14.3
module load openmpi/4.0.5

source .env/bin/activate

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show | grep inet | grep 192.168.205 | head -1 | cut -d " " -f 6 | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($SLURM_JOBID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=4
NODE_TYPE="a100"

NUM_NODES=1
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

# training config
SEQ_LENGTH=4096
DATA_PARALLEL_SIZE=$NUM_GPUS

MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=64

# optimizer config
LR=2e-5
MIN_LR=6.6e-7
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# checkpoint & tokenizer
TOKENIZER_MODEL=/home/kazuki/hf_checkpoints/Mistral-7B-v0.1/tokenizer.model
CHECKPOINT_DIR=/home/kazuki/hf_checkpoints/Mistral-7B-v0.1
CHECKPOINT_SAVE_DIR="/home/kazuki/checkpoints/Mistral-7b/instruction"

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data path
TRAIN_DATA_PATH="/home/kazuki/datasets/instruction/llm-jp-gpt4-self-instruct/train_data.jsonl"
VALID_DATA_PATH="/home/kazuki/datasets/instruction/llm-jp-gpt4-self-instruct/val_data.jsonl"

# job name
JOB_NAME="Mistral-7b-instruction-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"

# run

# tokenizer-type, tokenizer-model, train-iters are not used when instruction tuning

mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
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
  --sharding-strategy FULL_SHARD \
  --checkpoint-type LOCAL_STATE_DICT \
  --fsdp-activation-checkpointing \
  --instruction-tuning \
  --save-sampler-state \
  --use-mpi \
  --wandb-entity "okoge" \
  --wandb-project "kotoba-recipes-debug" \
  --wandb-name "${JOB_NAME}"
