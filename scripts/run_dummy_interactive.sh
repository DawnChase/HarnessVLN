#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
AGENT_CONFIG="config/agents/passthrough.yaml"
ENV_CONFIG="config/envs/dummy.yaml"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"

python -m harness.cli env \
  --environment "$ENV_CONFIG" \
  --agent "$AGENT_CONFIG"
