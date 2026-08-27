#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_CONFIG="config/benches/r2r_ce.yaml"
AGENT_CONFIG="config/agents/passthrough.yaml"
ENV_CONFIG="config/envs/habitat_r2r.yaml"
DUAL_ENV_CONFIG="config/envs/habitat_r2r_dualvln.yaml"
VLN_CONFIG="config/vln/dualvln.yaml"
RUN_CONFIG="config/runs/r2r_dualvln.yaml"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"

python -m harness.cli \
  "$BENCH_CONFIG" "$AGENT_CONFIG" "$ENV_CONFIG" "$DUAL_ENV_CONFIG" \
  "$VLN_CONFIG" "$RUN_CONFIG"
