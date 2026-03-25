#!/bin/bash

#SBATCH --job-name=run_fods_small_25
#SBATCH --output=run_fods_small_25_%j.out
#SBATCH --error=run_fods_small_25_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1

python ./eval_fods.py --data-source dipy --dipy-dataset small_25 --output-dir ../../outputs/fods_small_25
