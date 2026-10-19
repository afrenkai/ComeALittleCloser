# Batch execution

Submit from the project root; scripts use SLURM_SUBMIT_DIR instead of a fixed checkout path. Create outputs/ before submission. Partition, memory, CPU, and GPU requests are examples for the existing Slurm environment.

GPU scripts explicitly disable the default CPU dependency group, select the Ampere extra, and pass --ampere to the inference command. They use .venv-ampere; CPU checks and evaluation use the default .venv. Every uv command synchronizes its selected environment without replacing the other backend's dependencies.

The embedding jobs are embed.sbatch (Qwen3/BGE-M3/E5) and remaining-models.sbatch (Gemma/CLAP). location-prepare.sbatch prepares the frozen ablations and queries; location-evaluate.sbatch evaluates them after an afterok dependency. Transfer completed artifacts for local reporting. New runs must preserve the source/plan provenance they actually execute.

Turing is computation-only for this project. Research SSH sessions for submission, status, and file transfer must disable forwarding:

```bash
ssh -o ClearAllForwardings=yes -o ControlMaster=no -o ControlPath=none turing
```

Research web services and forwarding tunnels remain disabled. web.sbatch refuses execution. Leave the existing desktop-managed Codex SSH control connection enabled. The local explorer does not depend on a remote web service.
