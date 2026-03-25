#!/bin/bash

#SBATCH --job-name=b3s_eval
#SBATCH --output=b3s_eval%j.out
#SBATCH --error=b3s_eval%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1

#bash confusion_b3s_mlp_baseline.sh
#bash confusion_all_mlp_baseline.sh
bash confusion_all_conv_mlp_baseline.sh
