#!/bin/bash
#SBATCH --job-name=eval_sbi
#SBATCH --output=eval_sbi_%j.out
#SBATCH --error=eval_sbi_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1


SBI_CHECKPOINT="/home/macke/mgloeckler90/dmri/figures/b3s_baseline/runs/main_b3s/checkpoints/best.pt"
OUTPUT_JSON="sbi_eval.json"

export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu

python eval_sbi_checkpoint.py \
    --checkpoint "${SBI_CHECKPOINT}" \
    --output-json "${OUTPUT_JSON}" \
    --num-trials 512
