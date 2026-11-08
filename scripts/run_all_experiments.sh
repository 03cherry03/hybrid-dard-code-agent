#!/usr/bin/env bash
# The Python supervisor owns stage failures, process cleanup and email delivery.
set -euo pipefail
task_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_repo_dir"
exec python3 scripts/all_experiments.py "$@"
