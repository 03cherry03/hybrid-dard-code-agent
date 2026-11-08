#!/usr/bin/env python3
"""Plan or explicitly prepare native SWE repositories and validated environments."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_repository_preparation import (file_hash, make_plan, prepare_one,
                                         verify_prepared_environment, write_json)
from storage import data_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["swebench_verified", "swebench_multilingual"], required=True)
    parser.add_argument("--data-root", type=Path, default=data_root())
    parser.add_argument("--limit", type=int, default=1, help="0 prepares every task; failures never silently shrink selection")
    parser.add_argument("--instance-ids", default="")
    parser.add_argument("--prepare", action="store_true", help="Clone and install dependencies; default prints plans only")
    parser.add_argument("--harness-python", type=Path)
    parser.add_argument("--conda", default=os.environ.get("CONDA_EXE") or shutil.which("conda"))
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--test-timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.limit < 0 or args.timeout <= 0 or args.test_timeout <= 0:
        parser.error("Limits must be nonnegative and timeouts positive")
    root = args.data_root.resolve(); dataset = root / "datasets" / args.suite
    manifest = json.loads((dataset / "manifest.json").read_text())
    sealed = dataset / "sealed.jsonl"
    if file_hash(sealed) != manifest["sealed_sha256"]:
        raise ValueError("Sealed dataset checksum mismatch")
    rows = [json.loads(line) for line in sealed.read_text().splitlines() if line]
    if len(rows) != manifest["count"]:
        raise ValueError("Repository dataset count changed")
    wanted = set(args.instance_ids.split(",")) if args.instance_ids else set()
    if wanted:
        rows = [row for row in rows if row["instance_id"] in wanted]
        if {row["instance_id"] for row in rows} != wanted:
            raise ValueError("Unknown requested instance ID")
    rows = rows[:args.limit] if args.limit else rows
    if not args.prepare:
        print(json.dumps({"suite": args.suite, "count": len(rows), "prepare": False,
                          "plans": [make_plan(row, args.suite, root) for row in rows]}, indent=2))
        return 0
    if not args.conda:
        parser.error("conda is required for isolated task prefixes; activate the prepared control environment")
    harness_python = args.harness_python or root / "envs" / "swebench" / "bin" / "python"
    from repository_runner import verify_harness
    verify_harness(harness_python)
    lock_path = dataset / "repositories.lock.json"
    with (dataset / ".native-preparation.lock").open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks = json.loads(lock_path.read_text()) if lock_path.exists() else {}
        for index, row in enumerate(rows, 1):
            key = row["instance_id"]
            if key in locks:
                verify_prepared_environment(locks[key])
                print(json.dumps({"instance_id": key, "status": "already_ready", "index": index, "count": len(rows)}), flush=True)
                continue
            print(json.dumps({"instance_id": key, "status": "preparing", "index": index, "count": len(rows)}), flush=True)
            locks[key] = prepare_one(row, args.suite, root, harness_python=harness_python,
                                    conda=args.conda, timeout=args.timeout, test_timeout=args.test_timeout)
            write_json(lock_path, locks)
            print(json.dumps({"instance_id": key, "status": "ready", "environment_id": locks[key]["environment_id"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
