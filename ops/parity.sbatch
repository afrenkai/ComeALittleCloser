#!/bin/bash
#SBATCH --job-name=layer-parity
#SBATCH --partition=gtml
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/parity-%A_%a.log

# Our pooled output layer vs the model's Sentence Transformers embedding; one task per model.
# Each argument is "model|pooling|prefix", e.g. "nemotron-8b|mean|passage: ".
# Submit: sbatch -A gtml --array=0-N ops/parity.sbatch "<spec0>" "<spec1>" ...
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-ampere"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
specs=("$@")
IFS='|' read -r model_id pooling prefix <<< "${specs[$SLURM_ARRAY_TASK_ID]}"
uv run --locked --offline --no-group cpu --extra ampere -m come_a_little_bit_closer.models.layer_parity --ampere --model "$model_id" --pooling "$pooling" --prefix "$prefix"
