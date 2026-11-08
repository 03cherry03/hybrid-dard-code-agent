#!/usr/bin/env python3
"""Cache one pinned checkpoint under the campaign guard; no installs or manifest replacement."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.download_models import MODELS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--model", choices=tuple(MODELS), required=True)
    args = p.parse_args()
    from huggingface_hub import snapshot_download
    print(json.dumps({"stage": "download_start", "model": args.model,
                      "revision": MODELS[args.model]}), flush=True)
    snapshot = snapshot_download(repo_id=args.model, revision=MODELS[args.model],
        cache_dir=args.data_root / "cache/huggingface/hub",
        allow_patterns=["*.safetensors", "*.json", "*.py", "*.txt", "*.model", "*.tiktoken", "LICENSE*", "README.md"])
    print(json.dumps({"stage": "download_complete", "model": args.model, "snapshot": snapshot}), flush=True)


if __name__ == "__main__":
    main()
