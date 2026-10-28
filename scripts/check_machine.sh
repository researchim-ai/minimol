#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-src}"
PY="${PYTHON:-python3}"
if [[ -x .venv/bin/python ]]; then
  PY=".venv/bin/python"
fi
"$PY" -m minimol.cli env-check
