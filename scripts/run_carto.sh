#!/usr/bin/env bash

#SBATCH --job-name=noise-filter
#SBATCH --mail-user=sachdeva@ukp.informatik.tu-darmstadt.de
#SBATCH --output=/storage/ukp/work/sachdeva/job-%j
#SBATCH --mail-type=ALL
#SBATCH --time=72:00:00
#SBATCH --partition=ukp
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --gpus=1
#SBATCH --constraint="gpu_mem:80gb"

if [ -f .env ]; then
  export $(echo $(cat .env | sed 's/#.*//g'| xargs) | envsubst)
fi

BASE_PATH="/storage/ukp/work/sachdeva/research_projects/exp_calibration/src"

#CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/cartography/dynamics_filtering.py --filter --model_dir roberta-squad-t5-squad-cfs-cartography/ --metric variability --data_dir ${BASE_PATH}/data/squad/t5_squad_counterfactuals/ --filtering_output_dir ./roberta-squad-t5-squad-cfs-amb/
#CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/cartography/dynamics_filtering.py --filter --model_dir roberta-squad-t5-squad-cfs-cartography/ --metric confidence --data_dir ${BASE_PATH}/data/squad/t5_squad_counterfactuals/ --filtering_output_dir ./roberta-squad-t5-squad-cfs-hard/

CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/cf_generation/baseline_generation/cartography/run_train.py \
--model_name "roberta-base" \
--cf_path "src/data/squad/t5_squad_counterfactuals/rag_counterfactuals_complete_noise_min_filtered_final_dedup_1.jsonl" \
--seed 42 \
--output_dir "roberta-squad-t5-squad-cfs-carto-seed-42-new"