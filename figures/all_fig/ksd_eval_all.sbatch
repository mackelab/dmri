#!/bin/bash

#SBATCH --job-name=ksd_eval
#SBATCH --output=ksd_eval%j.out
#SBATCH --error=ksd_eval_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1

python ./ksd_eval_all.py --synthetic-samples 500_000 --real-samples 500_000
