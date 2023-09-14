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
if [ -f .env ]; then
  export $(echo $(cat .env | sed 's/#.*//g'| xargs) | envsubst)
fi

BASE_PATH="/storage/ukp/work/sachdeva/research_projects/exp_calibration/src"

MODEL="flan-ul2-context-rel-noise"

# List of dataset names
datasets=(
#"squad" \
#"squad_adversarial" \
#"trivia_qa" \
#"hotpot_qa" \
#"natural_questions" \
#"bioasq"
"news_qa" \
)

# List of seeds
seeds=(42 0 1)

# Function to calculate average
function calculate_average {
    total=0
    for score in "$@"
    do
        total=$(echo "scale=2; $total + $score" | bc)
    done
    average=$(echo "scale=2; $total / $#" | bc)
    echo $average
}

# Loop through the datasets
for DATASET_NAME in "${datasets[@]}"
do
    echo "Running scoring for dataset: ${DATASET_NAME}"

    # Loop through the seeds for each dataset
    for SEED in "${seeds[@]}"
    do
        MODEL_NAME="roberta-squad-${MODEL}-seed-${SEED}"

        echo "Running scoring for model: ${MODEL_NAME} with dataset: ${DATASET_NAME}"
        scores=($(CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/shortcuts/evaluate.py --model_name "${MODEL_NAME}" --dataset "${DATASET_NAME}"))
        echo "Finished scoring for model: ${MODEL_NAME} with dataset: ${DATASET_NAME}"
        echo "EM Score: ${scores[0]}"
        echo "F1 Score: ${scores[1]}"
        echo "-----------------------------------------"

        # Store the scores for each seed in separate arrays
        em_scores_array+=(${scores[0]})
        f1_scores_array+=(${scores[1]})
    done

    # Calculate the average EM and F1 scores for the dataset and model
    avg_em_score=$(calculate_average "${em_scores_array[@]}")
    avg_f1_score=$(calculate_average "${f1_scores_array[@]}")
    echo "Average EM Score for dataset: ${DATASET_NAME} and model: ${MODEL_NAME} is: ${avg_em_score}"
    echo "Average F1 Score for dataset: ${DATASET_NAME} and model: ${MODEL_NAME} is: ${avg_f1_score}"
    echo "========================================="

    # Clear the scores arrays for the next dataset
    em_scores_array=()
    f1_scores_array=()
done
