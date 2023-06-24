#!/usr/bin/env bash

#SBATCH --job-name=explainer
#SBATCH --mail-user=anon
#SBATCH --output=/xyz-storage-1/anon/job-%j
#SBATCH --mail-type=ALL
#SBATCH --time=72:00:00
#SBATCH --partition=xyz
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_mem:80gb"

if [ -f .env ]; then
  export $(echo $(cat .env | sed 's/#.*//g'| xargs) | envsubst)
fi

BASE_PATH="/storage/xyz/work/anon/research_projects/exp_calibration"


CUDA_LAUNCH_BLOCKING=1 python -u ${BASE_PATH}/src/rag/shap/run_shap.py
