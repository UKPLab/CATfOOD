##!/usr/bin/env bash
#
##SBATCH --job-name=exp
##SBATCH --mail-user=sachdeva@ukp.informatik.tu-darmstadt.de
##SBATCH --output=/ukp-storage-1/sachdeva/job-%j
##SBATCH --mail-type=ALL
##SBATCH --time=72:00:00
##SBATCH --partition=ukp
##SBATCH --cpus-per-task=4
##SBATCH --ntasks=1
##SBATCH --mem=32GB
##SBATCH --gpus=1
##SBATCH --constraint="gpu_mem:80gb"
#
if [ -f .env ]; then
  export $(echo $(cat .env | sed 's/#.*//g'| xargs) | envsubst)
fi

BASE_PATH="/ukp-storage-1/sachdeva/research_projects/exp_calibration/src"

# List of model names
models=(
#"roberta-squad-llama2-gpt-jt-cfs-seed-42" \
#"roberta-squad-llama2-llama-13b-cfs-seed-42" \
"roberta-squad-llama2-gpt-neox-cfs-seed-42" \
#"roberta-squad-llama2-flan-ul2-cfs-seed-42" \
)


# Loop through the model names
for MODEL_NAME in "${models[@]}"
do
    echo "Running exp. for model: ${MODEL_NAME}"
    export MODEL_NAME_OR_PATH=${MODEL_NAME}
    export DATASET="squad_adversarial"
    model_type=$(echo "$MODEL_NAME" | cut -d '-' -f 4)
    export INTERP_DIR="exp_llama2_$model_type"
    CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/calibration/explainers/shap/run_shap.py

    echo "Finished exp. for model: ${MODEL_NAME}"
    echo "--------------------------------------"
done
