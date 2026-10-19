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
