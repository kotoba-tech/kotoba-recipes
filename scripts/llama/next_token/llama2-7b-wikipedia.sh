#!/bin/bash
#$ -l rt_AF=2
#$ -l h_rt=3:00:00
#$ -j y
#$ -o outputs/llama/7b/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# switch virtual env
source .env/bin/activate

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile

if [[ "$SGE_RESOURCE_TYPE" == "rt_F" ]]; then
  export NUM_GPU_PER_NODE=4
  NODE_TYPE="v100"
elif [[ "$SGE_RESOURCE_TYPE" == "rt_AF" ]]; then
  export NUM_GPU_PER_NODE=8
  NODE_TYPE="a100"
else
  echo "Unrecognized SGE_RESOURCE_TYPE: $SGE_RESOURCE_TYPE"
fi

NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r line; do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done <"$SGE_JOB_HOSTLIST" >"$HOSTFILE_NAME"

# debugging flag
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

# training settings
NUM_EPOCHS=1

# batch size
BATCH_SIZE=8
GLOBAL_BATCH_SIZE=1024
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_SIZE * NUM_GPUS)))

if (($GRADIENT_ACCUMULATION_STEPS < 1)); then
  echo "Error: Gradient Accumulation Steps is less than 1. Exiting."
  exit 1
fi

# optimizer
LR=1e-4
LR_MIN=1e-5
LR_DECAY=0.80
LR_WARMUP=0.05
LR_DECAY_STYLE="cosine"
WEIGHT_DECAY=0.1

# seed
SEED=42

# dataset
NUM_WORKERS_DATALOADER=2

# checkpoint path
CHECKPOINTS_PATH=/path/to/llama/checkpoints/llama-2-7b-gbs_${GLOBAL_BATCH_SIZE}-${NODE_TYPE}_${NHOSTS}
mkdir -p $CHECKPOINTS_PATH

export CUDA_LAUNCH_BLOCKING=1

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none -map-by slot \
  -x PATH \
  python examples/finetuning.py \
  --enable_fsdp \
  --low_cpu_fsdp \
  --peft_method None \
  --mixed_precision \
  --pure_bf16 \
  --num_epochs $NUM_EPOCHS \
  --model_name /path/to/llama2/Llama-2-7b-hf \
  --tokenizer_name /path/to/llama2/Llama-2-7b-hf \
  --batch_size_training $BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --lr $LR \
  --lr_min $LR_MIN \
  --lr_warmup $LR_WARMUP \
  --lr_decay $LR_DECAY \
  --lr_decay_style $LR_DECAY_STYLE \
  --weight_decay $WEIGHT_DECAY \
  --fsdp_activation_checkpointing \
  --seed $SEED \
  --dataset "wikipedia_dataset" \
  --train_data_path "/path/to/datasets/llama2-llm-jp-corpus/v1.0.2/sample/ja_wiki/merged_train_0.jsonl" \
  --context_size 4096 \
  --num_workers_dataloader $NUM_WORKERS_DATALOADER \
  --save_model \
  --save_optimizer \
  --save_interval_iteration 100 \
  --save_checkpoint_path $CHECKPOINTS_PATH \
  --load_checkpoint_path $CHECKPOINTS_PATH \
  --use_mpi \
  --use_fast_kernels \
  --wandb_entity "okoge" \
  --wandb_project "llama-recipes-oss" \
  --wandb_name "llama2-7b-gbs_${GLOBAL_BATCH_SIZE}-lr_${LR}-lrmin_${LR_MIN}-wikipedia_dataset"
