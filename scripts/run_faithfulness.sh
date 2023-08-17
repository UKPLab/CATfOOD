##!/usr/bin/env bash
#
##SBATCH --job-name=eval
##SBATCH --mail-user=sachdeva@ukp.informatik.tu-darmstadt.de
##SBATCH --output=/ukp-storage-1/sachdeva/job-%j
##SBATCH --mail-type=ALL
##SBATCH --time=6:30:00
##SBATCH --partition=yolo
##SBATCH --qos=yolo
##SBATCH --nodelist=penelope
##SBATCH --cpus-per-task=4
##SBATCH --ntasks=1
##SBATCH --mem=64GB
##SBATCH --gres=gpu:1
##SBATCH --constraint="gpu_mem:80gb"
#
#if [ -f .env ]; then
#  export $(echo $(cat .env | sed 's/#.*//g'| xargs) | envsubst)
#fi

BASE_PATH="/storage/ukp/work/sachdeva/research_projects/exp_calibration/src"

# List of model names
models=(
"roberta-squad-flan-ul2-context-rel-noise-seed-42"  \
#"roberta-squad-gpt-neox-context-rel-seed-42"  \
#"roberta-squad-llama-context-rel-seed-42"  \
#"roberta-squad-t5-squad-cfs-seed-42"  \
#"roberta-squad" \
)

model_types=(
"flan_ul2_context_noise_rel"  \
#"gpt_neox_context_rel"  \
#"llama_context_rel"  \
#"rag"  \
#"base" \
)

datasets=(
#    "squad_adversarial"  \
#    "trivia_qa"  \
#    "hotpot_qa"  \
#    "news_qa"  \
#    "natural_questions"  \
    "bioasq"  \
    )

METRIC="comp"

num_models=${#models[@]}  # Get the number of models (assumes models and model_types have the same length)

for ((i = 0; i < num_models; i++))
do
    MODEL_NAME="${models[i]}"
    MODEL_TYPE="${model_types[i]}"

    echo "Running faithfulness exp. for model: ${MODEL_NAME}, model type: ${MODEL_TYPE}"

    for DATASET in "${datasets[@]}"
    do
        CUDA_LAUNCH_BLOCKING=1 python3 "${BASE_PATH}/faithfulness/faithfulness.py" \
            --metric "${METRIC}" \
            --model_name "${MODEL_NAME}" \
            --model_type "${MODEL_TYPE}" \
            --dataset "${DATASET}"  \
            --get_score
    done

    echo "Finished faithfulness exp. for model: ${MODEL_NAME}, model type: ${MODEL_TYPE}"
done
