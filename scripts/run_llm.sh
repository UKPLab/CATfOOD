#!/usr/bin/env bash

#SBATCH --job-name=train
#SBATCH --mail-user=anon
#SBATCH --output=/xyz-storage-1/anon/job-%j
#SBATCH --mail-type=ALL
#SBATCH --time=72:00:00
#SBATCH --partition=xyz
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_mem:48gb"

if [ -f .env ]; then
  export $(echo $(cat .env | sed 's/#.*//g'| xargs) | envsubst)
fi

BASE_PATH="/xyz-storage-1/anon/research_projects/exp_calibration/src"

#CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/few_shot/finetune_alpaca.py \
#    --base_model='decapoda-research/llama-13b-hf' \
#	  --data_path './src/few_shot/alpaca_data_cleaned.json' \
#    --num_epochs=10 \
#    --cutoff_len=512 \
#    --group_by_length \
#    --output_dir='./lora-alpaca-13b-10ep' \
#    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
#    --lora_r=16 \
#    --micro_batch_size=8


CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/few_shot/flan_ul2_qa.py --seed 0
CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/few_shot/flan_ul2_qa.py --seed 1