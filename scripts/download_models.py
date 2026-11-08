#!/usr/bin/env python3
"""Explicit, user-run download of the two frozen checkpoints; no installation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from storage import initialize, data_root

MODELS = {
    "Qwen/Qwen2.5-Coder-7B-Instruct": "c03e6d358207e414f1eca0bb1891e29f1db0e242",
    "apple/DiffuCoder-7B-Instruct": "4fdd4580064ca5d11808069ce78f88d068753c96",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=data_root())
    p.add_argument("--download", action="store_true", help="Required to download; otherwise prints the plan")
    args = p.parse_args()
    if not args.download:
        print(json.dumps({"models": MODELS, "data_root": str(args.data_root)}, indent=2))
        return
    root = initialize(args.data_root)
    from huggingface_hub import snapshot_download
    records = {}
    for repo, revision in MODELS.items():
        snapshot = Path(snapshot_download(repo_id=repo, revision=revision,
            cache_dir=root / "cache/huggingface/hub",
            allow_patterns=["*.safetensors", "*.json", "*.py", "*.txt", "*.model", "*.tiktoken", "LICENSE*", "README.md"]))
        alias = root / "models" / repo.split("/")[-1]
        if alias.is_symlink() and alias.resolve() == snapshot.resolve():
            pass
        elif alias.exists() or alias.is_symlink():
            raise RuntimeError(f"Refusing to replace existing model path: {alias}")
        else:
            alias.symlink_to(snapshot, target_is_directory=True)
        records[repo] = {"revision": revision, "snapshot": str(snapshot), "alias": str(alias)}
    (root / "models/manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print("Prepared both checkpoints. Use the recorded revisions and --local-files-only.")


if __name__ == "__main__":
    main()
