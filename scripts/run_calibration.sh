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
#"roberta-squad-flan-ul2-context-rel-noise-seed-42"  \
#"roberta-squad-gpt-neox-context-rel-seed-42"  \
#"roberta-squad-llama-context-rel-seed-42"  \
#"roberta-squad-t5-squad-cfs-seed-42"  \
"roberta-squad" \
)

datasets=(
#    "squad_adversarial"  \
#    "trivia_qa"  \
#    "hotpot_qa"  \
    "news_qa"  \
#    "natural_questions"  \
#    "bioasq"  \
)

exp_methods=(
#"attn"  \
#"sc_attn"  \
#"simple_grads"  \
#"ig"  \
"shap" \
)

for MODEL_NAME in "${models[@]}"
do
    for DATASET in "${datasets[@]}"
    do
        echo "Creating calibration dataset for model: ${MODEL_NAME}, dataset: ${DATASET}"
        for EXP_METHOD in "${exp_methods[@]}"
        do
            echo "Explanation method: ${EXP_METHOD}"
            CUDA_LAUNCH_BLOCKING=1 python3 "${BASE_PATH}/calibration/baseline/calibration_dataset.py" \
                --model_name "${MODEL_NAME}" \
                --dataset "${DATASET}"  \
                --method "${EXP_METHOD}"
        done
        echo "Finished creating calibration dataset for model: ${MODEL_NAME}, dataset: ${DATASET}"
    done
done
