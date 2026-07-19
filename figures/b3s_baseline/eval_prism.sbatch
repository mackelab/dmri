#!/bin/bash
#SBATCH --job-name=eval_prism
#SBATCH --output=eval_prism_%j.out
#SBATCH --error=eval_prism_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
PRISM_CONDA_ENV="${PRISM_CONDA_ENV:-}"
if [[ -n "${PRISM_CONDA_ENV}" ]]; then
    source "${CONDA_SH}"
    conda activate "${PRISM_CONDA_ENV}"
fi

PRISM_CHECKPOINT="${PRISM_CHECKPOINT:-/home/macke/mgloeckler90/dmri/results/updated5_b3s_2_4_6_128}"
OUTPUT_JSON="${OUTPUT_JSON:-${REPO_ROOT}/figures/b3s_baseline/eval_prism_${SLURM_JOB_ID}.json}"

python figures/b3s_baseline/eval_prism.py \
    --checkpoint "${PRISM_CHECKPOINT}" \
    --output-json "${OUTPUT_JSON}"
