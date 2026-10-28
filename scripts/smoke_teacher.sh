#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PYTHON:-python3}"
if [[ -x .venv/bin/python ]]; then PY=".venv/bin/python"; fi
"$PY" -m minimol.cli teacher carbon-a-smoke
"$PY" -m minimol.cli teacher carbon-x-smoke --with-metric 0.2 --without-metric 0.4
