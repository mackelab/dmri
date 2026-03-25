#!/bin/bash

#SBATCH --job-name=run_fods_data
#SBATCH --output=run_fods_data_%j.out
#SBATCH --error=run_fods_data_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1
#SBATCH --exclude=mlcbm004

bash eval_all_data2.sh
