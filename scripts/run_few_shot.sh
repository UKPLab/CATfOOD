#!/usr/bin/env bash

if [ -f .env ]; then
  export $(echo $(cat .env | sed 's/#.*//g'| xargs) | envsubst)
fi

BASE_PATH="/storage/ukp/work/sachdeva/research_projects/exp_calibration/src"

#CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/few_shot/flan_ul2_qa.py --seed 0
#CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/few_shot/flan_ul2_qa.py --seed 1
#CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/few_shot/flan_ul2_qa.py --seed 42
