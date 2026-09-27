#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

uv sync --frozen --all-groups
uv run --frozen pytest
uv run --frozen mypy code/main.py tests

rm -rf outputs/verify
uv run --frozen python code/main.py \
  --task-dir outputs/verify/tasks \
  --output outputs/verify/leaderboard.json \
  --include-per-example >/dev/null
test -s outputs/verify/skill-lm-eval-harness.md
uv run --frozen python - <<'PY'
import json
from pathlib import Path

board = json.loads(Path("outputs/verify/leaderboard.json").read_text(encoding="utf-8"))
assert board["schema"] == "lm-eval-harness/v1"
assert board["overall"] == 1.0
assert board["task_count"] == 5
assert all(task["score"] == 1.0 for task in board["tasks"].values())
assert all(len(task["examples"]) == task["total"] for task in board["tasks"].values())
PY
uv run --frozen python code/main.py >/dev/null
echo "verify: ok"
