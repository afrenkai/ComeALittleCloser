#!/bin/bash
#SBATCH --job-name=come-closer-gemma-clap
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/remaining-models-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-ampere"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TOKENIZERS_PARALLELISM=false
model_failures=0
for model_name in embeddinggemma clap; do
    if uv run --locked --no-group cpu --extra ampere -m come_a_little_bit_closer.models.batch --ampere --model "$model_name"; then
        printf '%s completed\n' "$model_name"
    else
        model_failures=$((model_failures + 1))
        printf '%s failed\n' "$model_name"
    fi
done
test "$model_failures" -eq 0
