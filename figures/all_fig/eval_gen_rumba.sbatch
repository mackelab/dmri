#!/bin/bash

#SBATCH --job-name=eval_gen_rumba
#SBATCH --output=eval_gen_rumba_%j.out
#SBATCH --error=eval_gen_rumba_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=h100-ferranti

python ./eval_generalization_errors.py \
  --no-eval-simformer \
  --eval-rumba \
  --output-json outputs/generalization_errors_all/rumba.json \
  --output-npz outputs/generalization_errors_all/rumba_mse.npz \
  --subsample-voxels 20_000 \
