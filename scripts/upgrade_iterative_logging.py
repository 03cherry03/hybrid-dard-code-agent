#!/usr/bin/env python3
"""Prepare and activate a reviewed logging-only source handoff.

This tool never stops/starts services, changes systemd units, or rewrites case
results. Preparation is safe while the experiment runs. Activation requires a
stopped service, idle assigned GPUs, and a receipt bound to the prepared bytes.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import difflib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iterative_tuning import file_sha, read_json

MIGRATION = "logging-migration-v2"
SNAPSHOT = "source-logging-v2"
PLAN_CHANGES = {"source", "source_sha256", "previous_plan"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_bytes(path, content):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def verify_plan(plan):
    # Keep the experiment's existing source/data/model/package checks intact.
    from scripts.iterative_experiments import verify_plan as verify
    verify(plan)


def source_manifest(root):
    root = Path(root).resolve(strict=True)
    result = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")
        for name in dirs:
            if (Path(directory) / name).is_symlink():
                raise ValueError("Symlinked source directory is unsupported")
        for name in sorted(files):
            path = Path(directory) / name
            if name.startswith(".") or not (path.suffix in (".py", ".json", ".sh")
                                             or name.startswith("requirements")):
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError("Source must contain regular files only")
            result[str(path.relative_to(root))] = file_sha(path)
    if "scripts/iterative_experiments.py" not in result:
        raise ValueError("Source tree lacks the iteration controller")
    return result


def retained_artifacts(output):
    """Inventory completed cases and their plans without following checkpoints."""
    output = Path(output)
    records = []
    for directory, dirs, files in os.walk(output, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and
                         not d.startswith("source") and d not in ("audit", "data", "__pycache__")
                         and not (Path(directory) / d).is_symlink())
        for name in sorted(files):
            path = Path(directory) / name
            relative = path.relative_to(output)
            if not (name in ("evaluation_plan.json", "binding.json", "summary.json", "results.json")
                    or ("cases" in relative.parts and path.suffix == ".json")):
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError("Evaluation artifacts must be regular files")
            # Writers publish these artifacts by atomic rename. Record one
            # opened version and reject an in-place write observed during read.
            with path.open("rb") as stream:
                before = os.fstat(stream.fileno())
                content = stream.read()
                after = os.fstat(stream.fileno())
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("Evaluation artifact changed during inventory")
            records.append({"path": str(relative), "size": len(content), "sha256": sha(content),
                            "kind": "case" if "cases" in relative.parts else "plan_or_result"})
    return sorted(records, key=lambda item: item["path"])


def migration_paths(output):
    output = Path(output).resolve(strict=True)
    return output, output / "audit" / MIGRATION, output / SNAPSHOT


@contextmanager
def migration_lock(output):
    audit = Path(output) / "audit"
    audit.mkdir(exist_ok=True)
    with (audit / (MIGRATION + ".lock")).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def validate_prepared(output):
    output, audit, snapshot = migration_paths(output)
    old_bytes = (audit / "previous_iteration_plan.json").read_bytes()
    new_bytes = (audit / "prepared_iteration_plan.json").read_bytes()
    old, new = json.loads(old_bytes), json.loads(new_bytes)
    receipt = read_json(audit / "migration_receipt.json")
    if (receipt["previous_plan_sha256"] != sha(old_bytes)
            or receipt["prepared_plan_sha256"] != sha(new_bytes)
            or receipt["source_diff_sha256"] != file_sha(audit / "source_diff.patch")
            or receipt["retained_inventory_sha256"] != file_sha(audit / "retained_artifacts.prepare.json")):
        raise ValueError("Prepared migration receipt binding mismatch")
    if {k: v for k, v in old.items() if k not in PLAN_CHANGES} != {
            k: v for k, v in new.items() if k not in PLAN_CHANGES}:
        raise ValueError("Migration changed experiment parameters")
    lineage = {"path": str(audit / "previous_iteration_plan.json"),
               "sha256": sha(old_bytes), "source": old["source"],
               "reason": "logging_instrumentation_v2"}
    if new["previous_plan"] != lineage or new["source"] != str(snapshot):
        raise ValueError("Migration lineage or source path mismatch")
    if new["output"] != str(output):
        raise ValueError("Migration output path mismatch")
    if source_manifest(snapshot) != new["source_sha256"]:
        raise ValueError("Prepared source inventory changed")
    verify_plan(old)
    verify_plan(new)
    return old_bytes, new_bytes, receipt


def prepare(output, source_root=ROOT, service="hybrid-dard-iterative-20261107.service"):
    output, audit, snapshot = migration_paths(output)
    original = output / "iteration_plan.json"
    if not original.is_file() or original.is_symlink():
        raise ValueError("Existing regular iteration_plan.json is required")
    with migration_lock(output):
        if audit.exists():
            old_bytes, _, receipt = validate_prepared(output)
            if original.read_bytes() != old_bytes:
                raise ValueError("Root plan no longer matches the prepared predecessor")
            return receipt
        if snapshot.exists() or snapshot.is_symlink():
            raise ValueError("Snapshot already exists without a complete preparation receipt")
        old_bytes = original.read_bytes()
        old = json.loads(old_bytes)
        if old["output"] != str(output):
            raise ValueError("Original plan belongs to another output")
        verify_plan(old)
        source_root = Path(source_root).resolve(strict=True)
        if source_root == output or output in source_root.parents or source_root in output.parents:
            raise ValueError("Source tree and experiment output must be separate")
        sources = source_manifest(source_root)
        temp_source = Path(tempfile.mkdtemp(prefix="." + SNAPSHOT + ".", dir=output))
        temp_audit = Path(tempfile.mkdtemp(prefix="." + MIGRATION + ".", dir=output / "audit"))
        try:
            for name, expected in sources.items():
                destination = temp_source / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                content = (source_root / name).read_bytes()
                if sha(content) != expected:
                    raise ValueError("Source changed during preparation")
                atomic_bytes(destination, content)
            if source_manifest(source_root) != sources:
                raise ValueError("Source changed during preparation")
            new = dict(old, source=str(snapshot), source_sha256=sources,
                       previous_plan={"path": str(audit / "previous_iteration_plan.json"),
                                      "sha256": sha(old_bytes), "source": old["source"],
                                      "reason": "logging_instrumentation_v2"})
            new_bytes = json_bytes(new)
            changes, patch = [], []
            for name in sorted(set(old["source_sha256"]) | set(sources)):
                if old["source_sha256"].get(name) == sources.get(name):
                    continue
                before = (Path(old["source"]) / name).read_text() if name in old["source_sha256"] else ""
                after = (temp_source / name).read_text() if name in sources else ""
                patch.extend(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                                 fromfile="previous/" + name, tofile="prepared/" + name))
                changes.append({"path": name, "previous_sha256": old["source_sha256"].get(name),
                                "prepared_sha256": sources.get(name)})
            patch_bytes = "".join(patch).encode()
            retained = retained_artifacts(output)
            retained_bytes = json_bytes(retained)
            receipt = {"schema_version": 1, "status": "prepared_pending_review", "prepared_unix": time.time(),
                       "service": service, "previous_plan_sha256": sha(old_bytes),
                       "prepared_plan_sha256": sha(new_bytes), "source_diff_sha256": sha(patch_bytes),
                       "retained_inventory_sha256": sha(retained_bytes),
                       "previous_case_count": sum(item["kind"] == "case" for item in retained),
                       "changed_files": changes, "experiment_parameters_unchanged": True,
                       "logging_only_semantics_requires_review": True}
            for name, content in (("previous_iteration_plan.json", old_bytes),
                                  ("prepared_iteration_plan.json", new_bytes), ("source_diff.patch", patch_bytes),
                                  ("retained_artifacts.prepare.json", retained_bytes),
                                  ("migration_receipt.json", json_bytes(receipt))):
                atomic_bytes(temp_audit / name, content)
            if original.read_bytes() != old_bytes:
                raise ValueError("Root plan changed during preparation")
            os.rename(temp_source, snapshot)
            sync_directory(output)
            os.rename(temp_audit, audit)
            sync_directory(audit.parent)
            return receipt
        finally:
            for temporary in (temp_source, temp_audit):
                if temporary.exists():
                    shutil.rmtree(temporary)


def require_idle(plan, service):
    result = subprocess.run(["systemctl", "--user", "show", service,
                            "--property=LoadState,ActiveState,SubState,MainPID"],
                            check=True, text=True, capture_output=True, timeout=10)
    state = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if state.get("LoadState") != "loaded" or state.get("ActiveState") != "inactive" or state.get("MainPID") != "0":
        raise RuntimeError("Main service must be loaded, inactive, and have no main process")
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                            check=True, text=True, capture_output=True, timeout=10)
    assigned = {gpu["uuid"] for gpu in plan["gpus"]}
    if any(line.split(",", 1)[0].strip() in assigned for line in result.stdout.splitlines() if line.strip()):
        raise RuntimeError("Assigned GPU still has a compute process")
    return state


def activate(output, validation_receipt=None):
    output, audit, _ = migration_paths(output)
    validation_receipt = Path(validation_receipt) if validation_receipt else output / "audit/prelaunch_logging_validation.json"
    with migration_lock(output), ExitStack() as stack:
        old_bytes, new_bytes, receipt = validate_prepared(output)
        plan = json.loads(new_bytes)
        validation_bytes = validation_receipt.read_bytes()
        validation = json.loads(validation_bytes)
        if validation.get("status") != "passed" or validation.get("prepared_plan_sha256") != sha(new_bytes):
            raise ValueError("Passed validation receipt for these exact prepared plan bytes is required")
        # These are the same advisory locks held by the running controller.
        lock_paths = [output / "iteration.lock"] + [Path(plan["data_root"]) / "work/gpu-locks" / (gpu["uuid"] + ".lock")
                                                       for gpu in sorted(plan["gpus"], key=lambda g: g["uuid"])]
        for path in lock_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = stack.enter_context(path.open("a"))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_idle(plan, receipt["service"])
        original = output / "iteration_plan.json"
        current = original.read_bytes()
        if current not in (old_bytes, new_bytes):
            raise ValueError("Root plan differs from both archived and prepared bytes")
        retained = retained_artifacts(output)
        actual = {item["path"]: item for item in retained}
        for item in read_json(audit / "retained_artifacts.prepare.json"):
            if actual.get(item["path"]) != item:
                raise ValueError("Previously completed case/plan/result changed before handoff")
        inventory_bytes = json_bytes(retained)
        activation = {"schema_version": 1, "status": "activation_intent", "activated_unix": time.time(),
                      "previous_plan_sha256": sha(old_bytes), "prepared_plan_sha256": sha(new_bytes),
                      "validation_receipt": str(validation_receipt.resolve()), "validation_sha256": sha(validation_bytes),
                      "retained_inventory_sha256": sha(inventory_bytes),
                      "previous_case_count": sum(item["kind"] == "case" for item in retained)}
        atomic_bytes(audit / "retained_artifacts.activation.json", inventory_bytes)
        atomic_bytes(audit / "activation_receipt.json", json_bytes(activation))
        # Recheck after the inventory and immediately before the one live write.
        require_idle(plan, receipt["service"])
        if original.read_bytes() != current:
            raise ValueError("Root plan changed before atomic activation")
        if current != new_bytes:
            atomic_bytes(original, new_bytes)
        activation["status"] = "activated"
        atomic_bytes(audit / "activation_receipt.json", json_bytes(activation))
        return activation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--service", default="hybrid-dard-iterative-20261107.service")
    parser.add_argument("--validation-receipt", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--activate", action="store_true")
    args = parser.parse_args(argv)
    result = prepare(args.output, args.source_root, args.service) if args.prepare else activate(args.output, args.validation_receipt)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
