#!/usr/bin/env bash

##SBATCH --job-name=attribution
##SBATCH --mail-user=anon
##SBATCH --output=/xyz-storage-1/anon/job-%j
##SBATCH --mail-type=ALL
##SBATCH --time=72:00:00
##SBATCH --partition=xyz
##SBATCH --cpus-per-task=4
##SBATCH --ntasks=1
##SBATCH --mem=64GB
##SBATCH --gpus=1
##SBATCH --constraint="gpu_mem:80gb"
#
#if [ -f .env ]; then
#  export $(echo $(cat .env | sed 's/#.*//g'| xargs) | envsubst)
#fi

BASE_PATH="/xyz-storage-1/anon/research_projects/exp_calibration/src"

#CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/simple_grads.py
#python ${BASE_PATH}/faithfulness.py --metric suff --model_name roberta-squad-flan-ul2-context-rel-noise-seed-42 --model_type flan_ul2_context_noise_rel --dataset squad_adversarial
#python ${BASE_PATH}/faithfulness.py --metric suff --model_name roberta-squad-gpt-neox-context-rel-seed-42 --model_type gpt_neox_context_rel --dataset squad_adversarial
#python ${BASE_PATH}/faithfulness.py --metric suff --model_name roberta-squad-llama-context-rel-seed-42 --model_type llama_context_rel --dataset squad_adversarial
#python ${BASE_PATH}/faithfulness.py --metric suff --model_name roberta-squad-t5-squad-cfs-seed-42 --model_type rag --dataset squad_adversarial
#
#python ${BASE_PATH}/faithfulness.py --metric comp --model_name roberta-squad-gpt-neox-context-rel-seed-42 --model_type gpt_neox_context_rel --dataset squad_adversarial
#python ${BASE_PATH}/faithfulness.py --metric comp --model_name roberta-squad-llama-context-rel-seed-42 --model_type llama_context_rel --dataset squad_adversarial
#python ${BASE_PATH}/faithfulness.py --metric comp --model_name roberta-squad-flan-ul2-context-rel-noise-seed-42 --model_type flan_ul2_context_noise_rel --dataset squad_adversarial
#python ${BASE_PATH}/faithfulness.py --metric comp --model_name roberta-squad-t5-squad-cfs-seed-42 --model_type rag --dataset squad_adversarial
#
#python ${BASE_PATH}/faithfulness_2.py --metric comp --model_name roberta-squad --model_type base --dataset squad_adversarial
#python ${BASE_PATH}/faithfulness_2.py --metric suff --model_name roberta-squad-gpt-neox-context-rel-seed-42 --model_type gpt_neox_context_rel --dataset trivia_qa
#python ${BASE_PATH}/faithfulness_2.py --metric suff --model_name roberta-squad-llama-context-rel-seed-42 --model_type llama_context_rel --dataset trivia_qa
#python ${BASE_PATH}/faithfulness_2.py --metric comp --model_name roberta-squad-flan-ul2-context-rel-noise-seed-42 --model_type flan_ul2_context_noise_rel --dataset squad_adversarial
#python ${BASE_PATH}/faithfulness_2.py --metric comp --model_name roberta-squad-t5-squad-cfs-seed-42 --model_type rag --dataset squad_adversarial

#python ${BASE_PATH}/faithfulness_2.py --metric comp --model_name roberta-squad-gpt-neox-context-rel-seed-42 --model_type gpt_neox_context_rel --dataset hotpot_qa
#python ${BASE_PATH}/faithfulness_2.py --metric comp --model_name roberta-squad-t5-squad-cfs-seed-42 --model_type rag --dataset hotpot_qa

#python ${BASE_PATH}/faithfulness_2.py --metric suff --model_name roberta-squad --model_type base --dataset squad_adversarial
#python ${BASE_PATH}/faithfulness_2.py --metric suff --model_name roberta-squad-gpt-neox-context-rel-seed-42 --model_type gpt_neox_context_rel --dataset squad_adversarial
#python ${BASE_PATH}/faithfulness_2.py --metric suff --model_name roberta-squad-llama-context-rel-seed-42 --model_type llama_context_rel --dataset squad_adversarial
#python ${BASE_PATH}/faithfulness_2.py --metric suff --model_name roberta-squad-flan-ul2-context-rel-noise-seed-42 --model_type flan_ul2_context_noise_rel --dataset squad_adversarial
#python ${BASE_PATH}/faithfulness_2.py --metric suff --model_name roberta-squad-t5-squad-cfs-seed-42 --model_type rag --dataset squad_adversarial

#python ${BASE_PATH}/faithfulness.py --metric comp --model_name roberta-squad --model_type base --dataset hotpot_qa
#python ${BASE_PATH}/faithfulness.py --metric comp --model_name roberta-squad-gpt-neox-context-rel-seed-42 --model_type gpt_neox_context_rel --dataset hotpot_qa
#python ${BASE_PATH}/faithfulness.py --metric comp --model_name roberta-squad-llama-context-rel-seed-42 --model_type llama_context_rel --dataset hotpot_qa
#python ${BASE_PATH}/faithfulness.py --metric comp --model_name roberta-squad-flan-ul2-context-rel-noise-seed-42 --model_type flan_ul2_context_noise_rel --dataset hotpot_qa
#python ${BASE_PATH}/faithfulness.py --metric comp --model_name roberta-squad-t5-squad-cfs-seed-42 --model_type rag --dataset hotpot_qa
#
#python ${BASE_PATH}/faithfulness.py --metric suff --model_name roberta-squad --model_type base --dataset hotpot_qa
#python ${BASE_PATH}/faithfulness.py --metric suff --model_name roberta-squad-gpt-neox-context-rel-seed-42 --model_type gpt_neox_context_rel --dataset hotpot_qa
#python ${BASE_PATH}/faithfulness.py --metric suff --model_name roberta-squad-llama-context-rel-seed-42 --model_type llama_context_rel --dataset hotpot_qa
#python ${BASE_PATH}/faithfulness.py --metric suff --model_name roberta-squad-flan-ul2-context-rel-noise-seed-42 --model_type flan_ul2_context_noise_rel --dataset hotpot_qa
#python ${BASE_PATH}/faithfulness.py --metric suff --model_name roberta-squad-t5-squad-cfs-seed-42 --model_type rag --dataset hotpot_qa

#python ${BASE_PATH}/faithfulness.py --metric suff --model_name roberta-squad --model_type base --dataset squad_adversarial
#python ${BASE_PATH}/faithfulness.py --metric suff --model_name roberta-squad-flan-ul2-context-rel-noise-seed-42 --model_type flan_ul2_context_noise_rel --dataset squad_adversarial
#python ${BASE_PATH}/faithfulness.py --metric suff --model_name roberta-squad-gpt-neox-context-rel-seed-42 --model_type gpt_neox_context_rel --dataset squad_adversarial
#python ${BASE_PATH}/faithfulness.py --metric suff --model_name roberta-squad-llama-context-rel-seed-42 --model_type llama_context_rel --dataset squad_adversarial
#python ${BASE_PATH}/faithfulness.py --metric suff --model_name roberta-squad-t5-squad-cfs-seed-42 --model_type rag --dataset squad_adversarial

#python ${BASE_PATH}/faithfulness.py --metric suff --model_name roberta-squad-t5-squad-cfs-seed-42 --model_type rag --dataset trivia_qa

python ${BASE_PATH}/faithfulness_2.py --metric suff --model_name roberta-squad-t5-squad-cfs-seed-42 --model_type rag --dataset trivia_qa
python ${BASE_PATH}/faithfulness_2.py --metric suff --model_name roberta-squad --model_type base --dataset trivia_qa
