#!/bin/bash

#SBATCH --job-name=run_coverages
#SBATCH --output=run_coverages_%j.out
#SBATCH --error=run_coverages_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1

# Run the coverage evaluation script
python ./ksd_eval_all.py 
