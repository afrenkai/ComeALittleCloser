#!/bin/bash
#SBATCH --job-name=come-closer-embed
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/embedding-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-ampere"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
uv sync --locked --no-group cpu --extra ampere
for model_name in qwen3 bge-m3 e5; do
    uv run --locked --no-group cpu --extra ampere -m come_a_little_bit_closer.models.batch --ampere --model "$model_name"
done
