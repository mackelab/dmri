#!/bin/bash

#SBATCH --job-name=run_fods_data_all
#SBATCH --output=run_fods_data_all_%j.out
#SBATCH --error=run_fods_data_all_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1

python ./eval_fods.py --methods all --batch-size 1000 --output-dir ../../outputs/fods_data2
