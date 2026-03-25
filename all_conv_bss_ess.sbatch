#!/bin/bash

#SBATCH --job-name=all_syn_eval
#SBATCH --output=all_syn_eval%j.out
#SBATCH --error=all_syn_eval%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1
#SBATCH --exclude=mlcbm005

dmri_eval +experiment=eval_all_conv_metrics_data_bss export.metrics.ess.options.voxel_subset_size=1000  export=ess_only export.metrics.ess.batch_size=10
