#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_CONFIG="config/benches/dummy.yaml"
RUN_CONFIG="config/runs/dummy_passthrough.yaml"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"

python -m harness.cli "$BASE_CONFIG" "$RUN_CONFIG"
