#!/usr/bin/env bash
# Frozen model revisions; all model/data downloads happen in explicit preparation.
set -euo pipefail
task_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$task_repo_dir/scripts/runtime_env.sh"
cd "$task_repo_dir"
exec "${PYTHON_BIN:-$HYBRID_DARD_ROOT/envs/inference/bin/python}" main.py \
  --data-root "$HYBRID_DARD_ROOT" --local-files-only \
  --ar-revision c03e6d358207e414f1eca0bb1891e29f1db0e242 \
  --revision 4fdd4580064ca5d11808069ce78f88d068753c96 "$@"
