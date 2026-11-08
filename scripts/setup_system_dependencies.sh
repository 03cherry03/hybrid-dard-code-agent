#!/usr/bin/env bash
# Explicit, user-run terminal installation of local system prerequisites.
set -euo pipefail
task_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$task_script_dir/runtime_env.sh"
exec python3 "$task_script_dir/system_dependencies.py" --install "$@"
