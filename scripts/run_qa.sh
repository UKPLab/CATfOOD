#!/usr/bin/env bash

#SBATCH --job-name=qa
#SBATCH --mail-user=sachdeva@ukp.informatik.tu-darmstadt.de
#SBATCH --output=/ukp-storage-1/sachdeva/job-%j
#SBATCH --mail-type=ALL
#SBATCH --time=72:00:00
#SBATCH --partition=ukp
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --gpus=1
#SBATCH --constraint="gpu_mem:16gb"

BASE_PATH="/ukp-storage-1/sachdeva/research_projects/exp_calibration"

if [ -f .env ]; then
  export $(echo $(cat .env | sed 's/#.*//g'| xargs) | envsubst)
fi

export SQUAD_DIR=outputs
ACTION=${1:-none}

if [ "$ACTION" = "train" ]; then
    dataset=$2
    exp_id=$3

    exp_prefix="exps/${dataset}_${exp_id}/"

    mkdir ${exp_prefix}
    cp run_qa.sh "${exp_prefix}/run_qa.sh"

    if [ "$dataset" = "squad" ]; then
        python -u run_qa.py \
            --model_type roberta \
            --model_name_or_path roberta-base \
            --dataset $dataset \
            --do_train \
            --do_eval \
            --disable_tqdm \
            --train_file $SQUAD_DIR/train_${dataset}.json \
            --predict_file $SQUAD_DIR/dev_${dataset}.json \
            --learning_rate 1.5e-5 \
            --weight_decay 0.1 \
            --evaluate_during_training \
            --num_train_epochs 4 \
            --overwrite_output_dir \
            --max_seq_length 512 \
            --logging_steps 250 \
            --eval_steps 2000 \
            --save_steps 2000 \
            --warmup_steps 1000 \
            --output_dir "${exp_prefix}output" \
            --per_gpu_train_batch_size 8 \
            --per_gpu_eval_batch_size 8 2>&1 | tee "${exp_prefix}log.txt"
    else
        echo "invalid dataset"
    fi

elif [ "$ACTION" = "eval" ]; then
    dataset=$2
    split=${3:-dev}

    python -u ${BASE_PATH}/src/calibration/run_qa.py \
        --model_type roberta \
        --model_name_or_path ${BASE_PATH}/roberta-squad \
        --tokenizer_name roberta-base \
        --dataset $dataset \
        --do_eval \
        --predict_file ${BASE_PATH}/src/data/${split}_${dataset}.json \
        --overwrite_output_dir \
        --max_seq_length 512 \
        --output_dir  ${BASE_PATH}/src/data/${dataset} \
        --per_gpu_eval_batch_size 100
else
    echo "train or eval"
fi