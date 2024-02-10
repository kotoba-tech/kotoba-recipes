#!/bin/bash
#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o outputs/tokenize/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# swich virtual env
source .env/bin/activate

DATASET_DIR=/bb/llm/gaf51275/llama/datasets/taishi-datasets/merged/en
OUTPUT_DIR=/bb/llm/gaf51275/llama/datasets/mistral-7b-ve/tokenized

mkdir -p ${OUTPUT_DIR}

# tokenize en books
python megatron_lm/tools/preprocess_data.py \
  --input ${DATASET_DIR}/en_books.jsonl \
  --output-prefix ${OUTPUT_DIR}/en_books \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model /bb/llm/gaf51275/llama/mistral/swallow-mistral-7B-v0.1-merged-tokenizer-nfkc-16k-hf/merged_tokenizer_sp/jalm_llama.model \
  --append-eod \
  --workers 64
