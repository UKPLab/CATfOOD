#!/usr/bin/env bash


if [ -f .env ]; then
  export $(echo $(cat .env | sed 's/#.*//g'| xargs) | envsubst)
fi

BASE_PATH="/storage/ukp/work/sachdeva/research_projects/exp_calibration/src"

#CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/rag/train_qg.py

CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/rag/train_qg.py \
--model_name "t5-large" \
--seed 42 \
--output_dir "t5-large-squad-qgen-seed-42"
