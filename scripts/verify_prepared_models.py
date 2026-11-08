#!/usr/bin/env python3
"""Read-only validation of pinned cached checkpoints and trusted integrity receipts."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.download_models import MODELS


def signature(path):
    path = Path(path)
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    if not resolved.is_file() or stat.st_size <= 0:
        raise ValueError("Checkpoint file is missing or empty: " + str(path))
    return {"path": str(path), "resolved": str(resolved), "size": stat.st_size,
            "device": stat.st_dev, "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns}


def manifest_identity(root):
    root = Path(root).resolve()
    path = root / "models/manifest.json"
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if set(manifest) != set(MODELS):
        raise ValueError("Prepared model manifest differs from the frozen checkpoint set")
    for repo, revision in MODELS.items():
        entry = manifest[repo]
        snapshot, alias = Path(entry["snapshot"]), Path(entry["alias"])
        if (entry["revision"] != revision or snapshot.name != revision
                or alias.resolve(strict=True) != snapshot.resolve(strict=True)):
            raise ValueError("Pinned checkpoint identity mismatch: " + repo)
    return manifest, hashlib.sha256(raw).hexdigest()


def _hash_file(path):
    before = signature(path)
    blob = Path(before["resolved"]).name
    if re.fullmatch(r"[a-f0-9]{64}", blob):
        digest, algorithm = hashlib.sha256(), "sha256"
    elif re.fullmatch(r"[a-f0-9]{40}", blob):
        digest, algorithm = hashlib.sha1(), "git_blob_sha1"
        digest.update(b"blob " + str(before["size"]).encode() + b"\0")
    else:
        raise ValueError("Checkpoint target has no verifiable content-addressed blob identity: " + str(path))
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != blob:
        raise ValueError("Checkpoint content hash mismatch: " + str(path))
    if signature(path) != before:
        raise ValueError("Checkpoint changed while verifying: " + str(path))
    return {**before, "hash_algorithm": algorithm, "expected_hash": blob, "actual_hash": actual}


def _check_shards(snapshot, index):
    from safetensors import safe_open
    weight_map = index["weight_map"]
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Empty safetensors weight map")
    records, total_bytes, total_tensors = [], 0, 0
    for filename in sorted(set(weight_map.values())):
        if Path(filename).name != filename:
            raise ValueError("Unexpected shard path in weight map")
        path = snapshot / filename
        before = signature(path)
        with path.open("rb") as stream:
            length_raw = stream.read(8)
            if len(length_raw) != 8:
                raise ValueError("Truncated safetensors header")
            header_size = struct.unpack("<Q", length_raw)[0]
            if header_size > 64 * 1024 * 1024 or header_size + 8 > before["size"]:
                raise ValueError("Invalid safetensors header size")
            header = json.loads(stream.read(header_size))
        expected = {key for key, target in weight_map.items() if target == filename}
        actual = set(header) - {"__metadata__"}
        if expected != actual:
            raise ValueError("Safetensors keys disagree with index: " + filename)
        payload = 0
        dtypes = set()
        with safe_open(str(path), framework="pt", device="cpu") as tensors:
            if set(tensors.keys()) != expected:
                raise ValueError("Safe tensor reader keys disagree with index")
            for key in expected:
                info = header[key]
                shape = info["shape"]
                if list(tensors.get_slice(key).get_shape()) != shape:
                    raise ValueError("Tensor shape mismatch: " + key)
                begin, end = info["data_offsets"]
                if not 0 <= begin <= end <= before["size"] - header_size - 8:
                    raise ValueError("Tensor data offset exceeds file")
                dtypes.add(info["dtype"])
                element_sizes = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8,
                                 "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
                if info["dtype"] not in element_sizes or math.prod(shape) * element_sizes[info["dtype"]] != end - begin:
                    raise ValueError("Tensor shape/dtype byte count mismatch: " + key)
                payload += end - begin
        if 8 + header_size + payload != before["size"]:
            raise ValueError("Safetensors file size and payload disagree")
        if signature(path) != before:
            raise ValueError("Checkpoint changed during header verification")
        records.append({"filename": filename, "tensor_count": len(expected),
                        "payload_bytes": payload, "file_bytes": before["size"], "dtypes": sorted(dtypes)})
        total_bytes += payload
        total_tensors += len(expected)
    if total_bytes != index["metadata"]["total_size"] or total_tensors != len(weight_map):
        raise ValueError("Total weight bytes/tensors disagree with shard index")
    return records


def verify_models(root, output):
    """Read every prepared checkpoint file once; never download or modify caches."""
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest, manifest_sha = manifest_identity(root)
    started = time.time()
    models, paths = {}, []
    for repo, entry in manifest.items():
        snapshot = Path(entry["snapshot"])
        for required in ("config.json", "tokenizer_config.json", "model.safetensors.index.json"):
            if not (snapshot / required).is_file():
                raise ValueError("Required checkpoint file missing: " + repo + "/" + required)
        if not (snapshot / "tokenizer.json").is_file() and not all(
                (snapshot / name).is_file() for name in ("vocab.json", "merges.txt")):
            raise ValueError("Checkpoint tokenizer data missing: " + repo)
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        shards = _check_shards(snapshot, index)
        files = sorted(path for path in snapshot.iterdir() if path.is_file())
        paths.extend(files)
        models[repo] = {**entry, "shards": shards, "file_count": len(files),
                        "tensor_count": len(index["weight_map"]), "payload_bytes": index["metadata"]["total_size"]}
        print(json.dumps({"model": repo, "header_validation": "passed", "shards": len(shards)}), flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = []
        for record in pool.map(_hash_file, paths):
            records.append(record)
            if record["size"] > 100 * 1024 * 1024:
                print(json.dumps({"verified": record["path"], "bytes": record["size"],
                                  "sha256": record["actual_hash"]}), flush=True)
    if manifest_identity(root)[1] != manifest_sha:
        raise ValueError("Model manifest changed during verification")
    report = {"schema_version": 1, "status": "verified", "root": str(root),
              "started_unix": started, "ended_unix": time.time(), "manifest_sha256": manifest_sha,
              "models": models, "files": records,
              "total_file_bytes": sum(record["size"] for record in records),
              "weight_shards": sum(len(model["shards"]) for model in models.values()),
              "verification": "Pinned revisions,aliases,index/header keys,shapes,dtypes,byte sizes and content hashes"}
    from scripts.all_experiments import write_json
    write_json(output, report)
    return report


def verify_receipt(root, receipt_path, expected_receipt_sha256):
    """Revalidate trusted hash evidence without rereading the 30GB payloads."""
    raw = Path(receipt_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_receipt_sha256:
        raise ValueError("Prepared model integrity receipt changed")
    report = json.loads(raw)
    manifest, manifest_sha = manifest_identity(root)
    if (report.get("schema_version") != 1 or report.get("status") != "verified"
            or report.get("root") != str(Path(root).resolve()) or report.get("manifest_sha256") != manifest_sha
            or set(report.get("models", {})) != set(MODELS)):
        raise ValueError("Prepared model integrity receipt does not match current manifest")
    expected_paths = {str(path) for entry in manifest.values()
                      for path in Path(entry["snapshot"]).iterdir() if path.is_file()}
    files = report.get("files", [])
    if len(files) != len(expected_paths) or {row.get("path") for row in files} != expected_paths:
        raise ValueError("Prepared model integrity receipt has incomplete file coverage")
    for row in files:
        current = signature(row["path"])
        if any(row.get(key) != value for key, value in current.items()):
            raise ValueError("Prepared checkpoint file changed after integrity verification: " + row["path"])
        if row.get("actual_hash") != row.get("expected_hash") or row.get("actual_hash") != Path(row["resolved"]).name:
            raise ValueError("Prepared checkpoint lacks matching content hash evidence")
    return {"status": "verified", "receipt_sha256": expected_receipt_sha256,
            "files": len(files), "weight_shards": report["weight_shards"],
            "original_verification_unix": report["ended_unix"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"status": verify_models(args.data_root, args.output)["status"]}), flush=True)


if __name__ == "__main__":
    main()
