#!/usr/bin/env bash
# Smoke training only. Full 35-epoch runs stay on the make targets.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
make data-reg-sample data-cds-sample data-admet-sample
make train-reg-smoke
echo "smoke training finished"
