#!/usr/bin/env bash

#SBATCH --job-name=s-realm
#SBATCH --mail-user=sachdeva@ukp.informatik.tu-darmstadt.de
#SBATCH --output=/ukp-storage-1/sachdeva/job-%j
#SBATCH --mail-type=ALL
#SBATCH --time=72:00:00
#SBATCH --partition=ukp
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH --constraint="gpu_mem:32gb"

if [ -f .env ]; then
  export $(echo $(cat .env | sed 's/#.*//g'| xargs) | envsubst)
fi

BASE_PATH="/storage/ukp/work/sachdeva/research_projects/exp_calibration/src"
CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/rag/pipeline.py
