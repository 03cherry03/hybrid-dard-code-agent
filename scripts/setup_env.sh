#!/usr/bin/env bash
# User-run local setup only. Every environment and cache lives under /data.
set -euo pipefail
task_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_repo_dir="$(cd -- "$task_script_dir/.." && pwd)"
task_target="${1:-help}"
case "$task_target" in
  inference|swebench|function|evaluators|functions|repositories|all) ;;
  *) cat <<'USAGE'
Usage: bash scripts/setup_env.sh inference|function|evaluators|functions|swebench|repositories|all
inference: Python 3.11, torch 2.7.1 CUDA 11.8, Transformers 4.51.3.
function: local Python/Node/Rust/C/C++/Fortran and native build toolchain.
swebench: Python 3.11 native preparation and pinned official log parsers.
evaluators: local EvalPlus (Python 3.11).
functions: inference + native function toolchain + EvalPlus; no repository prerequisites.
repositories: inference + native toolchain + SWE parser/controller and prerequisite checks.
all: functions + repositories. No models, datasets or target repositories.
No changes to base, user permissions, or system services. No sudo is invoked.
USAGE
     exit 0 ;;
esac
source "$task_script_dir/runtime_env.sh"
export PIP_CONFIG_FILE=/dev/null PIP_USER=false
unset PIP_TARGET PIP_PREFIX PIP_ROOT
[[ "$HYBRID_DARD_ROOT" == /data/* ]] || { printf '%s\n' 'Data root must be below /data.' >&2; exit 2; }
command -v conda >/dev/null || { printf '%s\n' 'conda is required.' >&2; exit 1; }
mkdir -p "$HYBRID_DARD_ROOT"/{datasets,envs,cache,models,results,work} "$TMPDIR" \
         "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HYBRID_DARD_ROOT/envs/locks"
if [[ "$task_target" == all || "$task_target" == repositories || "$task_target" == swebench ]]; then
  python3 "$task_script_dir/system_dependencies.py" --data-root "$HYBRID_DARD_ROOT" --check \
    --output "$HYBRID_DARD_ROOT/envs/locks/system.json"
fi

create_python_env() {
  local task_env_prefix="$HYBRID_DARD_ROOT/envs/$1" task_python_version="${2:-3.11.11}"
  if [[ ! -x "$task_env_prefix/bin/python" ]]; then
    conda create --yes --prefix "$task_env_prefix" --override-channels -c conda-forge "python=$task_python_version" pip
  fi
  "$task_env_prefix/bin/python" -I -c 'import sys; assert ".".join(map(str,sys.version_info[:2])) == sys.argv[1]' "${task_python_version%.*}"
}

freeze_env() {
  local task_name="$1"
  "$HYBRID_DARD_ROOT/envs/$task_name/bin/python" -m pip freeze --all > "$HYBRID_DARD_ROOT/envs/locks/$task_name.freeze.txt"
  conda list --prefix "$HYBRID_DARD_ROOT/envs/$task_name" --explicit > "$HYBRID_DARD_ROOT/envs/locks/$task_name.conda.txt"
}

setup_inference() {
  create_python_env inference
  local task_python="$HYBRID_DARD_ROOT/envs/inference/bin/python"
  "$task_python" -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
  "$task_python" -m pip install -r "$task_repo_dir/requirements.inference.txt"
  "$task_python" -m pip check
  "$task_python" -I -c 'import torch,transformers; assert torch.__version__=="2.7.1+cu118"; assert transformers.__version__=="4.51.3"'
  freeze_env inference
}

setup_function() {
  create_python_env function
  conda install --yes --prefix "$HYBRID_DARD_ROOT/envs/function" --override-channels -c conda-forge \
    'nodejs=22' 'rust=1.85' c-compiler cxx-compiler fortran-compiler git curl make cmake ninja \
    pkg-config autoconf automake libtool patch unzip zip
  freeze_env function
}

setup_swebench() {
  create_python_env swebench
  local task_python="$HYBRID_DARD_ROOT/envs/swebench/bin/python"
  "$task_python" -m pip install -r "$task_repo_dir/requirements.swebench.txt"
  "$task_python" -m pip install --no-deps 'swebench @ git+https://github.com/SWE-bench/SWE-bench.git@87ab1f6ced28f75ba73ca899dc759b019310944a'
  "$task_python" -I -c 'import sys; sys.path.insert(0,sys.argv[1]); from local_swe_worker import load_official_api; print(load_official_api()["identity"])' "$task_repo_dir"
  freeze_env swebench
}

setup_evaluators() {
  create_python_env evalplus
  local task_python="$HYBRID_DARD_ROOT/envs/evalplus/bin/python"
  "$task_python" -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
  "$task_python" -m pip install -r "$task_repo_dir/requirements.evalplus.txt"
  "$task_python" -m pip check
  "$task_python" -I -c 'import importlib.metadata as m; import evalplus.evaluate; assert m.version("evalplus")=="0.3.1"'
  freeze_env evalplus

}

case "$task_target" in
  inference) setup_inference ;;
  function) setup_function ;;
  swebench) setup_swebench ;;
  evaluators) setup_evaluators ;;
  functions) setup_inference; setup_function; setup_evaluators ;;
  repositories) setup_inference; setup_function; setup_swebench ;;
  all) setup_inference; setup_function; setup_swebench; setup_evaluators ;;
esac
printf 'Local setup complete: %s. Version records: %s/envs/locks\n' "$task_target" "$HYBRID_DARD_ROOT"
