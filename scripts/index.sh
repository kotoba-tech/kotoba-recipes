#!/bin/bash
#$ -l rt_F=1
#$ -l h_rt=1:0:00:00
#$ -j y
#$ -o outputs/index/
#$ -cwd

# swich virtual env
source .env/bin/activate

python src/llama_recipes/datasets/index.py
