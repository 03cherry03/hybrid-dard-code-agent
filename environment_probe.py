"""Read-only readiness report. Never installs, downloads, imports models, or allocates CUDA.

Run with the standard library: python environment_probe.py --data-root /data/hybrid-dard
Model completeness is structural (configuration/tokenizer/shards), not a weight checksum.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from prepare_data import SCOPES, selected_file, selected_suites, verify

PACKAGES = ("torch", "transformers", "accelerate", "bitsandbytes", "tokenizers",
            "safetensors", "huggingface_hub", "datasets", "evalplus", "swebench")
EXPECTED_INFERENCE = {"torch": "2.7.1+cu118", "transformers": "4.51.3"}
MODELS = ("Qwen/Qwen2.5-Coder-7B-Instruct", "apple/DiffuCoder-7B-Instruct")
ENVIRONMENTS = {"functions": ("inference", "function", "evalplus"),
                "repositories": ("inference", "function", "swebench"),
                "all": ("inference", "function", "evalplus", "swebench")}


def run_readonly(argv: list[str], timeout: float = 8) -> dict:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        return {"ok": result.returncode == 0, "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()[:3000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}


def installed_versions() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def disk_status(path: Path) -> dict:
    existing = path
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    usage = shutil.disk_usage(existing)
    return {"requested_path": str(path), "probed_path": str(existing),
            "total_gib": round(usage.total / 1024**3, 2),
            "free_gib": round(usage.free / 1024**3, 2),
            "existing_parent_writable": os.access(existing, os.W_OK)}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_safetensors_size(path: Path) -> None:
    """Validate declared tensor extent without reading or loading weight data."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors header: {path.name}")
        header_size = int.from_bytes(prefix, "little")
        if not 2 <= header_size <= min(100 * 1024 * 1024, size - 8):
            raise ValueError(f"Invalid safetensors header size: {path.name}")
        header = json.loads(stream.read(header_size))
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header: {path.name}")
    extents = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        offsets = tensor["data_offsets"]
        if len(offsets) != 2 or not all(isinstance(item, int) for item in offsets) or not 0 <= offsets[0] <= offsets[1]:
            raise ValueError(f"Invalid tensor extent: {path.name}")
        extents.append(offsets[1])
    if not extents or 8 + header_size + max(extents) != size:
        raise ValueError(f"Incomplete safetensors payload: {path.name}")


def inspect_model(path: Path) -> dict:
    result = {"path": str(path), "exists": path.is_dir(), "structurally_complete": False,
              "weight_integrity": "not_checked", "missing": [], "errors": []}
    if not path.is_dir():
        result["missing"] = ["model directory"]
        return result
    required = ["config.json", "tokenizer_config.json"]
    if not any((path / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")):
        result["missing"].append("tokenizer vocabulary")
    try:
        config = json.loads((path / "config.json").read_text())
        if not isinstance(config, dict):
            raise ValueError("Model configuration must be an object")
        result["model_type"] = config.get("model_type")
        result["resolved_revision"] = config.get("_commit_hash")
        if str(config.get("model_type", "")).lower() == "dream":
            required += ["modeling_dream.py", "configuration_dream.py", "generation_utils.py"]
        index_path = path / "model.safetensors.index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text())
            shards = sorted(set(index["weight_map"].values()))
            if not shards:
                raise ValueError("Empty safetensors shard index")
        elif (path / "model.safetensors").is_file():
            shards = ["model.safetensors"]
        else:
            shards = []
            result["missing"].append("model.safetensors or shard index")
        for shard in shards:
            if not isinstance(shard, str) or Path(shard).name != shard:
                raise ValueError("Unexpected shard path in model index")
        required += shards
        result["weight_shards"] = len(shards)
        result["weight_bytes"] = sum((path / name).stat().st_size for name in shards if (path / name).is_file())
        for name in shards:
            if (path / name).is_file():
                check_safetensors_size(path / name)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["errors"].append(str(exc))
    for name in required:
        entry = path / name
        if not entry.is_file() or entry.stat().st_size == 0:
            result["missing"].append(name)
    result["structurally_complete"] = not result["missing"] and not result["errors"]
    return result


def inspect_catalog(data_root: Path, verify_hashes: bool = True, *, scope: str | None = None) -> dict:
    if scope is not None:
        return inspect_scoped_catalog(data_root, scope=scope, verify_hashes=verify_hashes)
    path = data_root / "datasets" / "catalog.json"
    result = {"path": str(path), "exists": path.is_file(), "complete": False,
              "completeness_scope": "files_listed_in_catalog",
              "hashes_verified": verify_hashes, "files": [], "errors": []}
    if not path.is_file():
        return result
    try:
        catalog = json.loads(path.read_text())
        if not isinstance(catalog, dict):
            raise ValueError("Dataset catalog must be an object")
        result["catalog_sha256"] = file_sha256(path)
        result["datasets"] = catalog.get("datasets", {})
        records = catalog.get("files", [])
        if isinstance(records, dict):
            records = [{"path": key, **value} for key, value in records.items()]
        if not records:
            result["errors"].append("Catalog contains no file records")
        for record in records:
            item_path = Path(record["path"])
            candidate = item_path if item_path.is_absolute() else data_root / item_path
            candidate = candidate.resolve()
            if not candidate.is_relative_to(data_root.resolve()):
                raise ValueError("Catalog file is outside data root")
            expected = record.get("sha256")
            item = {"path": str(candidate), "exists": candidate.is_file(),
                    "sha256_expected": expected, "hash_matches": None}
            if candidate.is_file():
                item["size_bytes"] = candidate.stat().st_size
                expected_size = record.get("size_bytes", record.get("bytes"))
                item["size_matches"] = expected_size is None or item["size_bytes"] == expected_size
                if verify_hashes and expected:
                    item["hash_matches"] = file_sha256(candidate) == expected
            result["files"].append(item)
        result["complete"] = bool(records) and not result["errors"] and all(
            item["exists"] and item.get("size_matches") and item["sha256_expected"] and
            (item["hash_matches"] if verify_hashes else True) for item in result["files"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["errors"].append(str(exc))
    return result


def inspect_scoped_catalog(data_root: Path, *, scope: str, verify_hashes: bool = True) -> dict:
    suites = selected_suites(scope)
    path = data_root / "datasets/catalog.json"
    result = {"path": str(path), "exists": path.is_file(), "complete": False,
              "completeness_scope": scope, "hashes_verified": verify_hashes,
              "datasets": {}, "files": [], "errors": []}
    try:
        # The selected-suite verifier checks the selected entries in shared
        # manifests, so an excluded library suite cannot block this report.
        catalog = verify(data_root, scope, quiet=True, verify_hashes=verify_hashes)
        result["catalog_sha256"] = file_sha256(path)
        result["datasets"] = {suite: catalog["datasets"][suite] for suite in suites}
        result["files"] = [{"path": str(data_root / relative),
                            "validation": "selected_entries" if relative == "datasets/functions/manifest.json"
                            else "sha256_and_size" if verify_hashes else "size_only"}
                           for relative in catalog["files"] if selected_file(relative, scope)]
        result["complete"] = True
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        result["errors"].append(str(exc))
    return result


def probe(data_root: Path, *, model_paths: list[Path] | None = None, verify_hashes: bool = True,
          scope: str = "functions") -> dict:
    selected_suites(scope)
    gpu = run_readonly(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,uuid,pci.bus_id",
                        "--format=csv,noheader,nounits"])
    if gpu["ok"]:
        keys = ("index", "name", "driver_version", "memory_mib", "uuid", "pci_bus_id")
        gpu["devices"] = [dict(zip(keys, (value.strip() for value in row)))
                          for row in csv.reader(io.StringIO(gpu.pop("stdout")))]
    models = {}
    for model_name in MODELS:
        candidates = [data_root / "models" / model_name.split("/")[-1]]
        cache = data_root / "cache" / "huggingface" / "hub" / ("models--" + model_name.replace("/", "--"))
        if (cache / "snapshots").is_dir():
            candidates += sorted((cache / "snapshots").iterdir())
        models[model_name] = [inspect_model(item) for item in candidates]
    if model_paths:
        models["additional_paths"] = [inspect_model(path) for path in model_paths]
    environments = {}
    for name in ENVIRONMENTS[scope]:
        interpreter = data_root / "envs" / name / "bin" / "python"
        if interpreter.is_file():
            command = ("import importlib.metadata as m,json,sys; "
                       "names=" + repr(PACKAGES) + "; "
                       "d={x.metadata['Name'].lower().replace('-','_'):x.version for x in m.distributions()}; "
                       "print(json.dumps({'python':sys.version,'versions':{n:d.get(n.replace('-','_')) for n in names}}))")
            entry = run_readonly([str(interpreter), "-I", "-c", command])
            if entry["ok"]:
                entry["installed"] = json.loads(entry.pop("stdout"))
        else:
            entry = {"ok": False, "error": "environment not created"}
        environments[name] = {"python_path": str(interpreter), **entry}
    return {"read_only": True, "scope": scope, "data_root": str(data_root), "python": sys.version,
            "python_executable": sys.executable, "platform": platform.platform(),
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "installed_versions": installed_versions(), "expected_inference": EXPECTED_INFERENCE,
            "disk": disk_status(data_root), "gpu": gpu, "execution_backend": "local",
            "environments": environments, "models": models,
            "datasets": inspect_catalog(data_root, verify_hashes=verify_hashes, scope=scope)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("HYBRID_DARD_DATA_ROOT", os.environ.get("HYBRID_DARD_ROOT", "/data/hybrid-dard"))))
    parser.add_argument("--model-path", type=Path, action="append")
    parser.add_argument("--scope", choices=SCOPES, default="functions")
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()
    print(json.dumps(probe(args.data_root, model_paths=args.model_path,
                           verify_hashes=not args.skip_hashes, scope=args.scope), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
