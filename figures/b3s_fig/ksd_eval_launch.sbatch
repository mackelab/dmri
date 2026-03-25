#!/bin/bash

#SBATCH --job-name=b3s_eval
#SBATCH --output=b3s_eval%j.out
#SBATCH --error=b3s_eval%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1

python ksd_eval.py --synthetic-samples 500_000 --real-samples 500_000
