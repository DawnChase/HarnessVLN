#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/csl/Project/HarnessVLN"
PYTHON_BIN="python"
BASE_CONFIG="config/benches/dummy.yaml"
RUN_CONFIG="config/runs/dummy_passthrough.yaml"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"

"$PYTHON_BIN" -m harness.cli "$BASE_CONFIG" "$RUN_CONFIG"
