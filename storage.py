"""One external storage root; importing this module never creates directories."""
import os
from pathlib import Path


def data_root(value=None):
    return Path(value or os.environ.get("HYBRID_DARD_DATA_ROOT", "/data/hybrid-dard")).expanduser().resolve()


def initialize(root=None):
    root = data_root(root)
    for name in ("datasets", "datasets/function-cache", "datasets/functions/raw", "cache/huggingface",
                 "cache/pip", "cache/evalplus", "cache/bigcodebench", "envs", "models", "results", "work", "tmp"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root
