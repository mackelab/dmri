#!/bin/bash

#SBATCH --job-name=eval_all_new_data1
#SBATCH --output=eval_all_new_data1_%j.out
#SBATCH --error=eval_all_new_data1_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1
#SBATCH --exclude=mlcbm004,mlcbm009,mlcbm011,mlcbm007

python ./eval_all_new.py \
  --data-path ../../data/new_data \
  --output-dir outputs_new_data1 \
