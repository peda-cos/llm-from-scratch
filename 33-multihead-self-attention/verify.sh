#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

uv sync --frozen --all-groups
uv run --frozen pytest
uv run --frozen mypy main.py tests
uv run --frozen python main.py demo >/dev/null
uv run --frozen python main.py demo --heads 8 >/dev/null
echo "verify: ok"
