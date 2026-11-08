#!/usr/bin/env python3
"""Prepare frozen train/validation/holdout manifests from cached code tasks."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iterative_data import prepare_iterative_data


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("/data/hybrid-dard"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261105)
    p.add_argument("--results-root", type=Path)
    p.add_argument("--teacher-root", type=Path, action="append", dest="teacher_roots")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    result = prepare_iterative_data(args.data_root, args.output, args.seed,
                                   results_root=args.results_root, teacher_roots=args.teacher_roots)
    print(json.dumps({"output": str(args.output), "canonical_sources": result["canonical_sources"],
                      "deduplicated_tasks": result["deduplicated_tasks"], "splits": result["splits"],
                      "training_examples": result["training_examples"],
                      "missing_training_labels": len(result["missing_training_labels"]),
                      "proof": result["proof"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
