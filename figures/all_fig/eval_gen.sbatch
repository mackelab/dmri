#!/bin/bash

#SBATCH --job-name=eval_gen
#SBATCH --output=eval_gen_%j.out
#SBATCH --error=eval_gen_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1

python ./eval_generalization_errors.py \
  --mask b3s \
  --mask b3t \
  --mask all \
  --eval-rumba \
  --eval-csd \
  --output-json outputs/generalization_errors_12/summary.json \
  --output-npz outputs/generalization_errors_12/mse_per_voxel.npz \
  --slice-index 12
