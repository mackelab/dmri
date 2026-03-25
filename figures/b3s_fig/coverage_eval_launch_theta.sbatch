#!/bin/bash

#SBATCH --job-name=coverage_eval
#SBATCH --output=coverage_eval%j.out
#SBATCH --error=coverage_eval%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1

python coverage_eval.py --theta-synthetic-samples 10_000 --mode theta
