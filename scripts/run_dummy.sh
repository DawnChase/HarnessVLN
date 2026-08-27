#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_CONFIG="config/benches/dummy.yaml"
AGENT_CONFIG="config/agents/passthrough.yaml"
ENV_CONFIG="config/envs/dummy.yaml"
VLN_CONFIG="config/vln/dummy.yaml"
RUN_CONFIG="config/runs/dummy_passthrough.yaml"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"

python -m harness.cli \
  --benchmark "$BENCH_CONFIG" \
  --agent "$AGENT_CONFIG" \
  --environment "$ENV_CONFIG" \
  --vln "$VLN_CONFIG" \
  --run "$RUN_CONFIG"
