#!/usr/bin/env bash
# One-command setup and run. Linux + NVIDIA CUDA + Docker.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi
. .venv/bin/activate
python -m pip install --upgrade pip
if ! python -c 'import torch,sys; sys.exit(0 if torch.version.cuda else 1)' >/dev/null 2>&1; then
  # Change TORCH_INDEX_URL if server CUDA driver is older than cu128 requires.
  TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
  echo "[setup] Installing PyTorch from $TORCH_INDEX_URL"
  python -m pip install torch --index-url "$TORCH_INDEX_URL"
fi
python -m pip install -r requirements.txt
python -m pip check
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi
python -m unittest discover -s tests -v
RUNNER="${RUNNER:-docker}"
LIMIT="${LIMIT:-100}"
QUANT="${QUANT:-4bit}"
REPAIRS=1
if [ "$RUNNER" = "none" ]; then REPAIRS=0; fi
# Extra CLI args forwarded, e.g. bash run_server.sh --limit 10 --tokens 128
python main.py --suites humaneval,mbpp,multiple_rs --limit "$LIMIT" \
  --runner "$RUNNER" --repairs "$REPAIRS" --quant "$QUANT" \
  --decoders standard,dard --pull-images "$@"
