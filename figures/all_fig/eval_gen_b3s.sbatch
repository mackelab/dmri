#!/bin/bash

#SBATCH --job-name=eval_gen_b3s
#SBATCH --output=eval_gen_b3s_%j.out
#SBATCH --error=eval_gen_b3s_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1

python ./eval_generalization_errors.py \
  --mask b3s \
  --output-json outputs/generalization_errors_all/b3s.json \
  --output-npz outputs/generalization_errors_all/b3s_mse.npz \
  --num-samples 2 \
