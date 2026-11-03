#!/bin/bash
#SBATCH --job-name=come-closer-layers-probe
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/layers-probe-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
model_id="${1:?Pass the model ID as the first argument}"
mode="${2:-dev}"
uv run --locked --offline -m come_a_little_bit_closer.evaluation.layer_run --config experiments/layers-v1.json --model "$model_id" --mode "$mode"
