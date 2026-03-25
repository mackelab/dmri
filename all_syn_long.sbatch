#!/bin/bash

#SBATCH --job-name=all_syn_eval
#SBATCH --output=all_syn_eval%j.out
#SBATCH --error=all_syn_eval%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1

dmri_eval +experiment=eval_all_metrics_synthetic_long export.metrics.ess.batch_size=8 theta_sample.eval_batch_size=128
