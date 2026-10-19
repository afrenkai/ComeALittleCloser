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
- Compile the proposal from `paper/` and preserve eight pages including references.
- Do not claim annotation, audio collection, or model evaluation has happened unless there is an actual recorded run.
- Turing is computation-only for this project: SSH for batch jobs/status and file transfer is allowed with forwarding disabled. The user explicitly wants the existing Codex SSH control connection left enabled. Never run research web servers, forwarding tunnels, or other interactive application services there. Use cached pinned checkpoints with offline model loading. The web application runs locally from local artifacts. Keep `ops/web.sbatch` disabled.
