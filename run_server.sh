#!/usr/bin/env bash
# Launch the benchmark with the current environment's Python. No installation.
set -euo pipefail
cd "$(dirname "$0")"
BENCH_PYTHON="${PYTHON_BIN:-python}"
BENCH_RUNNER="${RUNNER:-local}"
BENCH_EFFECTIVE_RUNNER="$BENCH_RUNNER"
BENCH_EXPECT_RUNNER=0
for argument in "$@"; do
  if [ "$BENCH_EXPECT_RUNNER" = 1 ]; then
    BENCH_EFFECTIVE_RUNNER="$argument"
    BENCH_EXPECT_RUNNER=0
  elif [ "$argument" = "--runner" ]; then
    BENCH_EXPECT_RUNNER=1
  elif [[ "$argument" == --runner=* ]]; then
    BENCH_EFFECTIVE_RUNNER="${argument#--runner=}"
  fi
done
BENCH_REPAIRS="${REPAIRS:-1}"
if [ "$BENCH_EFFECTIVE_RUNNER" = "none" ]; then BENCH_REPAIRS=0; fi

# Additional CLI arguments are passed last and override these defaults.
# All tests execute in prepared local language/evaluator environments.
exec "$BENCH_PYTHON" main.py --limit "${LIMIT:-20}" --repeats "${REPEATS:-3}" \
  --quant "${QUANT:-bf16}" --device "${DEVICE:-cuda:0}" \
  --runner "$BENCH_RUNNER" --repairs "$BENCH_REPAIRS" "$@"
