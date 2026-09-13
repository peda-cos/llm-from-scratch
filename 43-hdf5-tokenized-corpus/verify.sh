#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

uv sync --frozen --all-groups
uv run --frozen pytest
uv run --frozen mypy code/main.py tests
uv run --frozen python code/main.py demo --workdir outputs/demo
uv run --frozen python code/main.py validate --index outputs/demo/shards.json
echo "verify: ok"
