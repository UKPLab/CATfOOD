#!/usr/bin/env bash

#SBATCH --job-name=ig-attribution
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

BASE_PATH="/ukp-storage-1/sachdeva/research_projects/exp_calibration/src"

# List of model names
models=(
#    "roberta-squad-flan-ul2-context-rel-noise-seed-42"  \
#    "roberta-squad-gpt-neox-context-rel-seed-42"  \
#    "roberta-squad-llama-context-rel-seed-42"  \
#    "roberta-squad-t5-squad-cfs-seed-42"  \
    "roberta-squad"  \
)

datasets=(
#    "squad_adversarial"  \
#    "trivia_qa"  \
#    "hotpot_qa"  \
    "news_qa"  \
    "natural_questions"  \
    "bioasq"  \
    )

# Loop through the model names
for MODEL_NAME in "${models[@]}"
do
    echo "Running exp. for model: ${MODEL_NAME}"

    for DATASET in "${datasets[@]}"
    do
        CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/calibration/explainers/attentions/attention.py --model_name "${MODEL_NAME}" --dataset "${DATASET}"
        CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/calibration/explainers/attentions/scaled_attention.py --model_name "${MODEL_NAME}" --dataset "${DATASET}"
        CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/calibration/explainers/gradients/simple_grads.py --model_name "${MODEL_NAME}" --dataset "${DATASET}"
        CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/calibration/explainers/gradients/integrated_grads.py --model_name "${MODEL_NAME}" --dataset "${DATASET}"
    done

    echo "Finished exp. for model: ${MODEL_NAME}"
    echo "--------------------------------------"
done