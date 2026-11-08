#!/usr/bin/env python3
"""Download one pinned optional candidate, independently of the baseline run."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.all_experiments import write_json
from scripts.verify_prepared_models import _check_shards, _hash_file
from experiment_notifications import notify, flush_pending
from tokenizer_contract import load_verified_tokenizer

MODEL = "apple/DiffuCoder-7B-cpGRPO"
REVISION = "98bc42bfd871eeec92fd0a8439ab34f75b405204"


def alert(output, stage, error_type):
    return notify(output, {"event_id": "candidate-model-" + stage,
        "stage": stage, "error_type": error_type, "output": str(output),
        "log_path": str(output / "candidate_download.log")})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--post-stop", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.post_stop:
        if os.environ.get("SERVICE_RESULT", "success") != "success":
            alert(args.output, "service-abnormal-exit", "CandidateDownloadServiceFailure")
        flush_pending(args.output, deadline_seconds=50)
        return 0
    def interrupted(signum, frame):
        raise InterruptedError("Candidate download interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    state_path = args.output / "candidate_download_state.json"
    started = time.time()
    from huggingface_hub import snapshot_download
    try:
        snapshot = None
        for attempt in range(1, 4):
            write_json(state_path, {"status": "downloading", "model": MODEL,
                "revision": REVISION, "attempt": attempt, "started_unix": started,
                "updated_unix": time.time()})
            try:
                snapshot = Path(snapshot_download(repo_id=MODEL, revision=REVISION,
                    cache_dir=args.data_root / "cache/huggingface/hub", max_workers=2,
                    allow_patterns=["*.safetensors", "*.json", "*.py", "*.txt",
                                    "*.model", "*.tiktoken", "LICENSE*", "README.md"]))
                break
            except Exception as exc:
                if isinstance(exc, InterruptedError):
                    raise
                write_json(state_path, {"status": "retrying", "attempt": attempt,
                    "error_type": type(exc).__name__, "updated_unix": time.time()})
                alert(args.output, "download-attempt-" + str(attempt), type(exc).__name__)
                if attempt == 3:
                    raise
                time.sleep(10)
        if snapshot.name != REVISION:
            raise ValueError("Candidate revision mismatch")
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        shards = _check_shards(snapshot, index)
        paths = sorted(p for p in snapshot.iterdir() if p.is_file())
        for required in ("config.json", "tokenizer_config.json", "modeling_dream.py"):
            if snapshot / required not in paths:
                raise ValueError("Candidate required file missing: " + required)
        write_json(state_path, {"status": "verifying", "updated_unix": time.time()})
        tokenizer, tokenizer_receipt = load_verified_tokenizer(
            snapshot, model_name=MODEL, revision=REVISION)
        del tokenizer
        with ThreadPoolExecutor(max_workers=2) as pool:
            files = list(pool.map(_hash_file, paths))
        receipt = {"schema_version": 1, "status": "verified", "model": MODEL,
            "revision": REVISION, "snapshot": str(snapshot), "files": files,
            "shards": shards, "started_unix": started, "ended_unix": time.time(),
            "tokenizer_contract": tokenizer_receipt,
            "verification": "Pinned revision, shard headers, content-addressed hashes and offline tokenizer contract"}
        write_json(args.output / "candidate_model_receipt.json", receipt)
        write_json(state_path, {"status": "completed", "model": MODEL,
            "revision": REVISION, "updated_unix": time.time()})
        print(json.dumps({"status": "verified", "files": len(files), "shards": len(shards)}), flush=True)
        return 0
    except BaseException as exc:
        write_json(state_path, {"status": "failed", "error_type": type(exc).__name__,
            "updated_unix": time.time()})
        alert(args.output, "preparation-failed", type(exc).__name__)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
