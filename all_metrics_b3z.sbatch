#!/bin/bash

#SBATCH --job-name=all_b3z_eval
#SBATCH --output=all_b3z_eval%j.out
#SBATCH --error=all_b3z_eval%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1

dmri_eval +experiment=eval_all_b3z_metrics_data export.metrics.ess.options.voxel_subset_size=20000
