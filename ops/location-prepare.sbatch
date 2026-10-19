#!/bin/bash
#SBATCH --job-name=come-closer-location-prepare
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/location-prepare-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-ampere"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
plan_path="${1:-experiments/location-v2.json}"
for model_name in qwen3 bge-m3 e5 embeddinggemma clap; do
    uv run --locked --offline --no-group cpu --extra ampere -m come_a_little_bit_closer.evaluation.location_prepare --ampere --config "$plan_path" --model "$model_name"
done
