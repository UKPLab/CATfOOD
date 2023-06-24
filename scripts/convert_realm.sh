#!/usr/bin/env bash

#SBATCH --job-name=convert
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


BASE_PATH="/ukp-storage-1/sachdeva/research_projects/exp_calibration/src"

CUDA_LAUNCH_BLOCKING=1 python3 ${BASE_PATH}/rag/convert.py --block_records_path "enwiki-20181220/blocks.tfr"  \
                                                           --block_emb_path "cc_news_pretrained/embedder/encoded/encoded.ckpt" \
                                                           --checkpoint_path "orqa_nq_model_from_realm/export/best_default/checkpoint/model.ckpt-300000"
