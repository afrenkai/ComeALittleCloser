#!/bin/bash
#SBATCH --job-name=come-closer-check
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:05:00
#SBATCH --output=outputs/check-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
uv run --locked ruff check src tests
uv run --locked black --check src tests
uv run --locked pytest -q
uv run --locked -m come_a_little_bit_closer.models.inspect > outputs/embedding-validation.json
