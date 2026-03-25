#!/bin/bash

#SBATCH --job-name=all_eval_mask
#SBATCH --output=all_eval_mask%j.out
#SBATCH --error=all_eval_mask%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1

python coverage_eval_conv.py --synthetic-samples 20_000 --batch-size 500 --mode mask --sequence long
