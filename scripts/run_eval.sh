##!/usr/bin/env bash
#
##SBATCH --job-name=eval
##SBATCH --mail-user=sachdeva@ukp.informatik.tu-darmstadt.de
##SBATCH --output=/ukp-storage-1/sachdeva/job-%j
##SBATCH --mail-type=ALL
##SBATCH --time=72:00:00
##SBATCH --partition=ukp
##SBATCH --cpus-per-task=4
##SBATCH --ntasks=1
##SBATCH --mem=32GB
##SBATCH --gres=gpu:1
##SBATCH --constraint="gpu_mem:32gb"
#
#if [ -f .env ]; then
#  export $(echo $(cat .env | sed 's/#.*//g'| xargs) | envsubst)
#fi

BASE_PATH="/ukp-storage-1/sachdeva/research_projects/exp_calibration/src"
#MODEL_NAME="roberta-squad-alpaca-13b-v1-temp-0.7-seed-1"
#MODEL_NAME="roberta-squad-llama-13b-v2-temp-0.7-seed-1"
#MODEL_NAME="roberta-squad-flan-t5-xxl-cn-filtered-cfs-seed-1"

#CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/shortcuts/inference.py --model_name ${MODEL_NAME} --dataset "squad"
#CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/shortcuts/inference.py --model_name ${MODEL_NAME} --dataset "squad_adversarial"
#CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/shortcuts/inference.py --model_name ${MODEL_NAME} --dataset "trivia_qa"
#CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/shortcuts/inference.py --model_name ${MODEL_NAME} --dataset "hotpot_qa"
##CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/shortcuts/inference.py --model_name ${MODEL_NAME} --dataset "news_qa"
#CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/shortcuts/inference.py --model_name ${MODEL_NAME} --dataset "natural_questions"
#CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/shortcuts/inference.py --model_name ${MODEL_NAME} --dataset "bioasq"

#python ${BASE_PATH}/faithfulness.py --metric comp --model_name roberta-squad-flan-ul2-context-rel-noise-seed-42 --model_type flan_ul2_context_noise_rel --dataset trivia_qa
#python ${BASE_PATH}/faithfulness.py --metric comp --model_name roberta-squad-gpt-neox-context-rel-seed-42 --model_type gpt_neox_context_rel --dataset trivia_qa
#python ${BASE_PATH}/faithfulness.py --metric comp --model_name roberta-squad-llama-context-rel-seed-42 --model_type llama_context_rel --dataset trivia_qa
#python ${BASE_PATH}/faithfulness.py --metric comp --model_name roberta-squad-t5-squad-cfs-seed-42 --model_type rag --dataset trivia_qa
#python ${BASE_PATH}/faithfulness.py --metric comp --model_name roberta-squad --model_type base --dataset trivia_qa

#python ${BASE_PATH}/faithfulness_2.py --metric suff --model_name roberta-squad-flan-ul2-context-rel-noise-seed-42 --model_type flan_ul2_context_noise_rel --dataset trivia_qa
#python ${BASE_PATH}/faithfulness_2.py --metric suff --model_name roberta-squad-t5-squad-cfs-seed-42 --model_type rag --dataset trivia_qa

#python ${BASE_PATH}/faithfulness.py --metric suff --model_name roberta-squad-flan-ul2-context-rel-noise-seed-42 --model_type flan_ul2_context_noise_rel --dataset hotpot_qa
#python ${BASE_PATH}/faithfulness.py --metric suff --model_name roberta-squad-gpt-neox-context-rel-seed-42 --model_type gpt_neox_context_rel --dataset hotpot_qa
#python ${BASE_PATH}/faithfulness.py --metric suff --model_name roberta-squad-llama-context-rel-seed-42 --model_type llama_context_rel --dataset hotpot_qa
#python ${BASE_PATH}/faithfulness.py --metric suff --model_name roberta-squad-t5-squad-cfs-seed-42 --model_type rag --dataset hotpot_qa
#python ${BASE_PATH}/faithfulness.py --metric suff --model_name roberta-squad --model_type base --dataset hotpot_qa

python ${BASE_PATH}/faithfulness_2.py --metric suff --model_name roberta-squad-flan-ul2-context-rel-noise-seed-42 --model_type flan_ul2_context_noise_rel --dataset trivia_qa
python ${BASE_PATH}/faithfulness_2.py --metric suff --model_name roberta-squad-gpt-neox-context-rel-seed-42 --model_type gpt_neox_context_rel --dataset trivia_qa
python ${BASE_PATH}/faithfulness_2.py --metric suff --model_name roberta-squad-llama-context-rel-seed-42 --model_type llama_context_rel --dataset trivia_qa
python ${BASE_PATH}/faithfulness_2.py --metric suff --model_name roberta-squad-t5-squad-cfs-seed-42 --model_type rag --dataset trivia_qa
python ${BASE_PATH}/faithfulness_2.py --metric suff --model_name roberta-squad --model_type base --dataset trivia_qa