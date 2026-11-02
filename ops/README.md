# Batch execution

Submit from the project root; scripts use SLURM_SUBMIT_DIR instead of a fixed checkout path. Create outputs/ before submission. Partition, memory, CPU, and GPU requests are examples for the existing Slurm environment.

GPU scripts explicitly disable the default CPU dependency group, select the Ampere extra, and pass --ampere to the inference command. They use .venv-ampere; CPU checks and evaluation use the default .venv. Every uv command synchronizes its selected environment without replacing the other backend's dependencies.

The embedding jobs are embed.sbatch (Qwen3/BGE-M3/E5) and remaining-models.sbatch (Gemma/CLAP). location-prepare.sbatch prepares the frozen ablations and queries; location-evaluate.sbatch evaluates them after an afterok dependency. Transfer completed artifacts for local reporting. New runs must preserve the source/plan provenance they actually execute.

Turing is computation-only for this project. Research SSH sessions for submission, status, and file transfer must disable forwarding:

```bash
ssh -o ClearAllForwardings=yes -o ControlMaster=no -o ControlPath=none turing
```

Research web services and forwarding tunnels remain disabled. web.sbatch refuses execution. Leave the existing desktop-managed Codex SSH control connection enabled. The local explorer does not depend on a remote web service.

## Boston College Andromeda

Work from the checkout under `/projects/weilab/$USER`. `/home` is a small filesystem shared by every user, so `ops/bc.env` keeps the uv cache, uv-managed Python, Hugging Face hub, and Torch cache under `/projects/weilab/$USER/.cache`. It also sets the Slurm account and emails `$USER@bc.edu` when a job begins, ends, or fails. Install uv once with `curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh`.

```bash
source ops/bc.env
sbatch ops/check.sbatch
bc_sbatch_gpu ops/embed.sbatch
bc_sbatch_gtml ops/embed.sbatch
```

The cu129 Torch wheels support sm_75 and newer, so `bc_sbatch_gpu` restricts GPU jobs to A100, L40S, and H200 nodes and skips the V100 nodes. When `short` is backed up, `bc_sbatch_gtml` submits to the gtml L40S node instead. The site profile sets `SBATCH_EXPORT=none`, so `bc.env` names the cache variables jobs receive; add any new variable a job needs to that list. Download pinned checkpoints on the login node with `hf download <checkpoint> --revision <revision>` before running the offline jobs.
