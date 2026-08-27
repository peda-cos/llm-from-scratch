#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

uv sync --frozen --all-groups
uv run --frozen pytest
uv run --frozen mypy code/main.py tests
uv run --frozen python code/main.py >/dev/null
uv run --frozen python code/main.py demo --layers 12 >/dev/null
echo "verify: ok"
