#!/bin/bash

#SBATCH --job-name=eval_gen_all_lam0p8
#SBATCH --output=eval_gen_all_lam0p8_%j.out
#SBATCH --error=eval_gen_all_lam0p8_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1

python ./eval_generalization_errors.py \
  --mask all0.8 \
  --output-json outputs/generalization_errors_all2/all_lam0p8.json \
  --output-npz outputs/generalization_errors_all2/all_lam0p8_mse.npz \
