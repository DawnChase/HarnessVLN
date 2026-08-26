#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/csl/Project/HarnessVLN"
CONDA_BIN="/home/csl/.local/anaconda3/bin/conda"
CONDA_ENV="harnessvln"
BENCH_CONFIG="config/benches/r2r_ce.yaml"
AGENT_CONFIG="config/agents/passthrough.yaml"
ENV_CONFIG="config/envs/habitat_r2r.yaml"
DUAL_ENV_CONFIG="config/envs/habitat_r2r_dualvln.yaml"
VLN_CONFIG="config/vln/dualvln.yaml"
RUN_CONFIG="config/runs/r2r_dualvln.yaml"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"

"$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" python -m harness.cli \
  "$BENCH_CONFIG" "$AGENT_CONFIG" "$ENV_CONFIG" "$DUAL_ENV_CONFIG" \
  "$VLN_CONFIG" "$RUN_CONFIG"
