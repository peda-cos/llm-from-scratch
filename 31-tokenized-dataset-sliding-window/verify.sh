#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

uv sync --frozen --all-groups
uv run --frozen pytest
uv run --frozen mypy main.py tests
uv run --frozen python main.py demo --epochs 1 >/dev/null
uv run --frozen python main.py demo --context-length 32 --epochs 1 >/dev/null
echo "verify: ok"
