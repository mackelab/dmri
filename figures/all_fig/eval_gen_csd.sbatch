#!/bin/bash

#SBATCH --job-name=eval_gen_csd
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=h100-ferranti

echo "Starting job"


python ./eval_generalization_errors.py \
  --no-eval-simformer \
  --eval-csd \
  --output-json outputs/generalization_errors_all/csd.json \
  --output-npz outputs/generalization_errors_all/csd_mse.npz

echo "Job completed"
