#!/bin/bash
#SBATCH --job-name=learn_with_score
#SBATCH --output=learn_with_score_%j.out
#SBATCH --error=learn_with_score_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=h100-ferranti

dmri_eval +experiment=eval_average_model_selection
