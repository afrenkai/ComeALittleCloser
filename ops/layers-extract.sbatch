#!/bin/bash
#SBATCH --job-name=come-closer-layers-extract
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/layers-extract-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-ampere"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
model_id="${1:?Pass the model ID as the first argument}"
plan_path="${2:-experiments/layers-v1.json}"
uv run --locked --offline --no-group cpu --extra ampere -m come_a_little_bit_closer.evaluation.layer_extract --ampere --config "$plan_path" --model "$model_id"
