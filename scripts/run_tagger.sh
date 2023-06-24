#!/usr/bin/env bash

#SBATCH --job-name=tagger
#SBATCH --mail-user=anon
#SBATCH --output=/storage/xyz/work/anon/job-%j
#SBATCH --mail-type=ALL
#SBATCH --time=72:00:00
#SBATCH --partition=xyz
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH --constraint="gpu_mem:32gb"

if [ -f .env ]; then
  export $(echo $(cat .env | sed 's/#.*//g'| xargs) | envsubst)
fi

BASE_PATH="/storage/xyz/work/anon/research_projects/exp_calibration/src"

CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/calibration/calib_exp/run_tagger.py --dataset squad