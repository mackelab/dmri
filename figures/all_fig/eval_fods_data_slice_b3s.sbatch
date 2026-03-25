#!/bin/bash

#SBATCH --job-name=run_fods_data
#SBATCH --output=run_fods_data_%j.out
#SBATCH --error=run_fods_data_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1

python ./eval_fods.py --output-dir ../../outputs/fods_data_slice5 --slice-axis z --slice-index 34 --batch-size 100 --theta-sample 5000 --methods b3s
