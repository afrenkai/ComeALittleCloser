@RTK.md

# Style Guide for Python

- `uv` is used for all projects. Run Python files with `uv run <file.py>` and modules with `uv run -m <module>`. Do not add Python shebangs or invoke Python scripts directly.
- No global consts. If one is necessary, put it into a relevant Hydra configuration; otherwise, put it into a protocol or schema.
- Do not use `zip`; use descriptive variable names.
- Prefer einops for PyTorch tensor transformations and contractions because its descriptive axis names make tensor logic easier to read.
- Add tests sparingly, only when they provide clear value. Keep them terse, prefer assertions where possible, and focus on edge cases.
- Formalize all formal mathematical statements in Lean 4.
- Organize code by domain, for example: `data`, `evaluation`, `models`, and `training`.
- Put Pydantic schemas, protocols, and type aliases in dedicated domain modules.
- Type every input and output. Use `jaxtyping` tensor shapes with `beartype` at numerical boundaries. This may not work with tools such as `torch.dynamo`; adhere as closely as possible where exceptions are required.
- Do not use inline imports in functions or methods. Keep imports at module scope, including optional-dependency boundaries. Do not use `getattr`, `Any`, `__all__`, or runtime path mutation.
- Prefer descriptive public names and short modules. Avoid generic iterator names, explanatory comments, and compatibility branches without an active contract.
- Omit one-line docstrings that only restate a function name or signature. Keep a docstring only when it records a substantive contract not clear from names and types.
- Verify Python changes with Ruff, Black, and focused tests.
- Prefer native, purpose-built tools over one-time Python or shell scripts whenever possible.

## Naming

- Underscores in normal variable names are allowed. Never use `_` as a throwaway variable or placeholder (for example, `for _, value in ...`); name every binding descriptively. This rule is language-agnostic.
- Do not use private methods or functions with leading underscores. All methods and functions must be public.

# Project contract

Follow the user's Python style rules and run Python through `uv`.

- CPU is the default. Select the `ampere` extra explicitly for GPU jobs and pass `--ampere` to inference commands. Keep environment synchronization enabled on every run.
- Use `master` as the base branch and create feature branches from it.

- Keep domain schemas and numerical type aliases in their domain's `schemas.py`.
- Keep raw snapshots and caches out of Git; preserve acquisition checksums.
- Distinguish coordinate controls, preprocessing observations, and learned-model results.
- Check Python changes with Ruff, Black, and focused tests.
- Formalize mathematical definitions and theorems in `formal/`; check with `lake build ComeCloser`.
- Compile the proposal from `paper/` and preserve eight pages including references. The full paper draft is `paper/final.tex`; on BC compile it with pdflatex and bibtex (latexmk is not installed).
- Do not claim annotation, audio collection, or model evaluation has happened unless there is an actual recorded run.
- Start here: `docs/PROGRESS.md` is the current state of the project (results, open issues, next steps, owners); `experiments/LAYERS_RUNS.md` is the authoritative run log. Read both before changing experiments.
- Held-out test discipline (layer probes): the test split was used once, under `experiments/layers-v7.json` (2026-11-04); its frozen claims and results are `docs/FREEZE.md` and `experiments/LAYERS_V7_CLAIMS.md`. Further analysis of those outputs is post hoc. `--mode final` runs once per model, only for cells frozen in the plan's `final_cells` and recorded in `experiments/LAYERS_RUNS.md` before the run. Never delete, move, or redirect outputs to rerun final mode, and never pick settings after seeing test results. The same held-out cities were then used once each for two more frozen claim sets:

- `experiments/reflected-v1.json`: claims in `docs/REFLECTED_FREEZE.md`, results in `experiments/REFLECTED_V1_RESULTS.md`.
- `experiments/fix-v1.json`: claims in `docs/FIX_FREEZE.md`, results in `experiments/FIX_V1_RESULTS.md`.

Their modules refuse a second final run. Test new claims on fresh held-out places rather than these cities again.
- `experiments/fresh-v1.json`: fresh GeoNames towns (5,000-14,999 people, not in `cities15000`). Its test part (80% of the towns) was used once on 2026-11-04 (claims in `docs/FRESH_FREEZE.md`, results in `experiments/FRESH_V1_RESULTS.md`). `fresh_places` refuses a second test run. Its check part may be rerun for pipeline checks.
- Frozen plans: once an `experiments/layers-v*.json` or `location-v*.json` plan has been used for a recorded run, do not edit it; create the next version. Record every job (including failures and cancellations) with its ID in the run log.
- BC Andromeda cluster (login node a002): keep login-node usage under 15% CPU and 10% memory. Anything heavier than editing, git, small unit tests, or job submission (probe runs, reports over real features, ridge fits, long tests) goes through Slurm. Run `source ops/bc.env` first; see `ops/README.md`. The `gtml` partition is usually idle but requires a GPU request even for CPU work (`--gres=gpu:l4:1`). Never `scontrol update` queued jobs; cancel and resubmit instead.
- Large artifacts are not in Git: extracted layer features (`data/processed/layers-v1/features`, about 370 GB) and outputs (`outputs/`) live on the cluster under `/projects/weilab/zhangdjr/ComeALittleCloser`. Gated checkpoints (gemma-2-9b) need a Hugging Face token at download time only; jobs load cached pinned revisions offline.
- Turing is computation-only for this project: SSH for batch jobs/status and file transfer is allowed with forwarding disabled. The user explicitly wants the existing Codex SSH control connection left enabled. Never run research web servers, forwarding tunnels, or other interactive application services there. Use cached pinned checkpoints with offline model loading. The web application runs locally from local artifacts. Keep `ops/web.sbatch` disabled.
