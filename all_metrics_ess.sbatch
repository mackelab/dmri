#!/bin/bash

#SBATCH --job-name=all_b3s_eval
#SBATCH --output=all_b3s_eval%j.out
#SBATCH --error=all_b3s_eval%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1

dmri_eval +experiment=eval_all_b3s_metrics_data export.metrics.ess.options.voxel_subset_size=20000 export=ess_only export.metrics.ess.batch_size=100
