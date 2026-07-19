#!/bin/bash

#SBATCH --job-name=train_sbi
#SBATCH --output=train_sbi_%j.out
#SBATCH --error=train_sbi_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1

python train_script_sbi.py --run-name "main_b3s" --wandb-mode "online" --num-steps 30_000_000 --sim-type "Ball3StickSharedDiffusivity"
