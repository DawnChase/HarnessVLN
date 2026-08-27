#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
AGENT_CONFIG="config/agents/passthrough_streamvln.yaml"
RUNNER_CONFIG="config/runners/r2r_streamvln.yaml"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"

python -m harness.cli run \
  --runner "$RUNNER_CONFIG" \
  --agent "$AGENT_CONFIG"
