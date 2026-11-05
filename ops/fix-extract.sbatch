#!/bin/bash
#SBATCH --job-name=fix-extract
#SBATCH --partition=gtml
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/fix-extract-%A_%a.log

# Instruction-conditioned extraction for a fix plan; one array task per model.
# Submit: sbatch -A gtml --array=0-2 ops/fix-extract.sbatch <plan> <model0> <model1> <model2>
# Fresh towns only: add --export=ALL,EXTRACT_ARGS="--towns plain-and-instructed" (or "--towns instructed").
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-ampere"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
plan_path="${1:?Pass the fix plan}"
shift
models=("$@")
model_id="${models[$SLURM_ARRAY_TASK_ID]}"
uv run --locked --offline --no-group cpu --extra ampere -m come_a_little_bit_closer.evaluation.instructed_extract --ampere --config "$plan_path" --model "$model_id" ${EXTRACT_ARGS:-}
