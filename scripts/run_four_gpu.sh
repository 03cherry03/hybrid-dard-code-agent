#!/usr/bin/env bash
# Prepare a frozen plan by default. --execute explicitly starts inference.
set -euo pipefail
task_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$task_repo_dir/scripts/runtime_env.sh"
cd "$task_repo_dir"
exec python3 scripts/four_gpu.py "$@"
