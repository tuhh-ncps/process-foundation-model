#!/usr/bin/env bash
# One-time environment setup for CONTAINERLESS clusters (e.g. oland, which has no
# Apptainer/Docker/Singularity/Podman). Installs uv (user-space, no root) and builds the
# project's .venv from pyproject.toml + uv.lock, then verifies torch/CUDA and the package.
#
# Run this ON the cluster, from the repo root (`bash scripts/setup_native.sh` or `make setup_native`).
# Idempotent — re-run after dependency changes. The .venv is per-machine and gitignored, so each
# cluster builds its own; it is never rsynced.
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"   # repo root

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  echo "[setup] installing uv (user-space)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
echo "[setup] uv $(uv --version)"

echo "[setup] uv sync --extra tracking   (builds .venv; downloads torch + deps on first run)"
uv sync --extra tracking

echo "[setup] verifying torch + CUDA + project import..."
PYTHONPATH=src uv run python - <<'PY'
import torch, pm_foundation  # noqa: F401
print(f"torch {torch.__version__}  cuda_avail={torch.cuda.is_available()}  ndev={torch.cuda.device_count()}")
print("pm_foundation import OK")
PY
echo "[setup] done — .venv ready. Submit jobs with:  make submit_oland EXPERIMENT=... DATASET=..."
