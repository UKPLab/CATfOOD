#!/usr/bin/env bash

##SBATCH --job-name=carto
##SBATCH --mail-user=anon
##SBATCH --output=/xyz-storage-1/anon/job-%j
##SBATCH --mail-type=ALL
##SBATCH --time=72:00:00
##SBATCH --partition=xyz
##SBATCH --cpus-per-task=4
##SBATCH --ntasks=1
##SBATCH --mem=32GB
##SBATCH --gpus=1
##SBATCH --constraint="gpu_mem:32gb"
#
#if [ -f .env ]; then
#  export $(echo $(cat .env | sed 's/#.*//g'| xargs) | envsubst)
#fi

BASE_PATH="/xyz-storage-1/anon/research_projects/exp_calibration/src"


############## LLM Counterfactuals ##############

#CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/rag/train_qa.py \
#--model_name "roberta-base" \
#--cf_path "src/data/squad/counterfactual_data_gpt_neox_20b_v2_qg_pipeline_all_data_cleaned.jsonl" \
#--seed 0 \
#--output_dir "roberta-squad-gpt-neox-v2-temp-0.7-seed-0"
#
#CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/rag/train_qa.py \
#--model_name "roberta-base" \
#--cf_path "src/data/squad/counterfactual_data_gpt_neox_20b_v2_qg_pipeline_all_data_cleaned.jsonl" \
#--seed 1 \
#--output_dir "roberta-squad-gpt-neox-v2-temp-0.7-seed-1"

#CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/rag/train_qa.py \
#--model_name "roberta-base" \
#--cf_path "src/data/squad/counterfactual_data_gpt_jt_v2_qg_pipeline_all_data_cleaned.jsonl" \
#--seed 0 \
#--output_dir "roberta-squad-gpt-jt-v2-temp-0.7-seed-0"
#
#CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/rag/train_qa.py \
#--model_name "roberta-base" \
#--cf_path "src/data/squad/counterfactual_data_gpt_jt_v2_qg_pipeline_all_data_cleaned.jsonl" \
#--seed 1 \
#--output_dir "roberta-squad-gpt-jt-v2-temp-0.7-seed-1"

#CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/rag/train_qa.py \
#--model_name "roberta-base" \
#--cf_path "src/data/squad/flan_ul2_collated_data_with_answers_processed_filtered.jsonl" \
#--seed 0 \
#--output_dir "roberta-squad-flan-ul2-filtered-seed-0"
#
#CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/rag/train_qa.py \
#--model_name "roberta-base" \
#--cf_path "src/data/squad/flan_ul2_collated_data_with_answers_processed_filtered.jsonl" \
#--seed 1 \
#--output_dir "roberta-squad-flan-ul2-filtered-seed-1"
#
#CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/rag/train_qa.py \
#--model_name "roberta-base" \
#--cf_path "src/data/squad/flan_ul2_collated_data_with_answers_processed.jsonl" \
#--seed 42 \
#--output_dir "roberta-squad-flan-ul2-v1-temp-0.7-seed-42"
#
#CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/rag/train_qa.py \
#--model_name "roberta-base" \
#--cf_path "src/data/squad/flan_ul2_collated_data_with_answers_processed_context_relevance_filtered.jsonl" \
#--seed 42 \
#--output_dir "roberta-squad-flan-ul2-context-rel-filtered-seed-42"
#
#CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/rag/train_qa.py \
#--model_name "roberta-base" \
#--cf_path "src/data/squad/flan_ul2_collated_data_with_answers_processed_context_relevance_filtered.jsonl" \
#--seed 0 \
#--output_dir "roberta-squad-flan-ul2-context-rel-filtered-seed-0"
#
#CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/rag/train_qa.py \
#--model_name "roberta-base" \
#--cf_path "src/data/squad/flan_ul2_collated_data_with_answers_processed_context_relevance_filtered.jsonl" \
#--seed 1 \
#--output_dir "roberta-squad-flan-ul2-context-rel-filtered-seed-1"


############ CARTOGRAPHY #############

CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/cf_generation/baseline_generation/train_qa.py \
--model_name "roberta-base" \
--cf_path "src/data/squad/t5_squad_counterfactuals/rag_counterfactuals_complete_noise_min_filtered_final_dedup_1.jsonl" \
--seed 42 \
--output_dir "roberta-squad-t5-squad-cfs-seed-42-new"

CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/cf_generation/baseline_generation/train_qa.py \
--model_name "roberta-base" \
--cf_path "src/data/squad/t5_squad_counterfactuals/rag_counterfactuals_complete_noise_min_filtered_final_dedup_1.jsonl" \
--seed 0 \
--output_dir "roberta-squad-t5-squad-cfs-seed-0-new"

CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/cf_generation/baseline_generation/train_qa.py \
--model_name "roberta-base" \
--cf_path "src/data/squad/t5_squad_counterfactuals/rag_counterfactuals_complete_noise_min_filtered_final_dedup_1.jsonl" \
--seed 1 \
--output_dir "roberta-squad-t5-squad-cfs-seed-1-new"