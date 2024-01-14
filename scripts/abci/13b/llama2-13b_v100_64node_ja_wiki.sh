#!/bin/bash
#$ -l rt_F=64
#$ -l h_rt=24:00:00
#$ -j y
#$ -o outputs/13b/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# swich virtual env
cd /home/acf15649kv/work/finetune/llama-recipes
source .env/bin/activate

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
NUM_NODES=$NHOSTS
NUM_GPU_PER_NODE=4
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

CHECKPOINTS_PATH=/groups/gaf51217/fujii/checkpoints/llama-2-13b/llama-recipies

# hugginface setting
export HF_HOME=/scratch/$(whoami)/.cache/huggingface/

mkdir -p $CHECKPOINTS_PATH

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
  --use_fp16 \
  --num_epochs 1 \
  --dataset "ja_wikipedia_dataset" \
  --model_name /groups/gaf51217/fujii/finetune/llama2/Llama-2-13b-hf \
  --batch_size 1 \
  --dist_checkpoint_root_folder $CHECKPOINTS_PATH \
  --dist_checkpoint_folder ja-wiki \
  --use_mpi \
  --use_fast_kernels \
  --wandb_name "llama2-13b_v100_ja_wiki"
