#!/usr/bin/env bash

#SBATCH --job-name=attn
#SBATCH --mail-user=anon
#SBATCH --output=/xyz-storage-1/anon/job-%j
#SBATCH --mail-type=ALL
#SBATCH --time=72:00:00
#SBATCH --partition=xyz
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH --constraint="gpu_mem:80gb"

if [ -f .env ]; then
  export $(echo $(cat .env | sed 's/#.*//g'| xargs) | envsubst)
fi

BASE_PATH="/xyz-storage-1/anon/research_projects/exp_calibration/src"

CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/attention.py
