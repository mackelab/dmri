#!/bin/bash

#SBATCH --job-name=eval_gen_dti
#SBATCH --output=eval_gen_dti_%j.out
#SBATCH --error=eval_gen_dti_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=h100-ferranti

python ./eval_generalization_errors.py \
  --no-eval-simformer \
  --eval-dti \
  --output-json outputs/generalization_errors_all/dti.json \
  --output-npz outputs/generalization_errors_all/dti_mse.npz
