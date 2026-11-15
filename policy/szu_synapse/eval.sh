#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK="${1:?task name required}"
CHECKPOINT="${2:?absolute checkpoint directory required}"
GPU="${3:-0}"
if (( $# >= 3 )); then shift 3; else shift 2; fi
export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$REPO_ROOT/policy/pi05/src:$REPO_ROOT/policy/pi05/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
"${PYTHON_BIN:-python}" scripts/eval_policy.py --config policy/szu_synapse/deploy_policy.yml \
    --overrides --task_name "$TASK" --checkpoint_path "$CHECKPOINT" "$@"
