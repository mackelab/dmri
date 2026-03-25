#!/bin/bash

#SBATCH --job-name=cov_eval_all_theta
#SBATCH --output=cov_eval_all_theta%j.out
#SBATCH --error=cov_eval_all_theta%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1

python ./cov_eval_all.py --theta-synthetic-samples 20_000 --mode theta --sequence both --theta-mask-hyper 0.95
