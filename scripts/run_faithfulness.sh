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

BASE_PATH="/storage/ukp/work/sachdeva/research_projects/exp_calibration/src"

# List of model names
models=(
"roberta-squad-llama2-gpt-jt-cfs-seed-42" \
"roberta-squad-llama2-llama-13b-cfs-seed-42" \
"roberta-squad-llama2-gpt-neox-cfs-seed-42" \
"roberta-squad-llama2-flan-ul2-cfs-seed-42" \
)

METRIC="suff"

# Loop through the model names
for MODEL_NAME in "${models[@]}"
do
    echo "Running faithfulness exp. for model: ${MODEL_NAME}"
    CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/faithfulness/faithfulness.py --metric "${METRIC}" --model_name "${MODEL_NAME}" --model_type "${MODEL_NAME}" --dataset "squad_adversarial" --get_score
    CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/faithfulness/faithfulness.py --metric "${METRIC}" --model_name "${MODEL_NAME}" --model_type "${MODEL_NAME}" --dataset "trivia_qa" --get_score
    CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/faithfulness/faithfulness.py --metric "${METRIC}" --model_name "${MODEL_NAME}" --model_type "${MODEL_NAME}" --dataset "hotpot_qa" --get_score
    echo "Finished faithfulness exp. for model: ${MODEL_NAME}"
done


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

#python ${BASE_PATH}/faithfulness_2.py --metric suff --model_name roberta-squad-flan-ul2-context-rel-noise-seed-42 --model_type flan_ul2_context_noise_rel --dataset trivia_qa
#python ${BASE_PATH}/faithfulness_2.py --metric suff --model_name roberta-squad-gpt-neox-context-rel-seed-42 --model_type gpt_neox_context_rel --dataset trivia_qa
#python ${BASE_PATH}/faithfulness_2.py --metric suff --model_name roberta-squad-llama-context-rel-seed-42 --model_type llama_context_rel --dataset trivia_qa
#python ${BASE_PATH}/faithfulness_2.py --metric suff --model_name roberta-squad-t5-squad-cfs-seed-42 --model_type rag --dataset trivia_qa
#python ${BASE_PATH}/faithfulness_2.py --metric suff --model_name roberta-squad --model_type base --dataset trivia_qa
