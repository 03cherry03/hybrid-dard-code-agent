#!/usr/bin/env python3
"""Merge verified pinned-AR TRAIN teachers without touching frozen source splits.

Generation and execution belong to iterative_eval_worker. This module only
validates their frozen receipts and adds synthetic passing implementations.
Validation/holdout source identities and tests never become training labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets_code import Task
from iterative_data import (PROMPT_VERSION, build_iterative_messages, fenced_completion,
                            normalize_source_id, source_from_identity, validate_training_examples)
from iterative_tuning import (AR, digest, file_sha, freeze_json, read_json, read_jsonl,
                              task_id, validate_candidate)


def _frozen_data(data_dir):
    data_dir = Path(data_dir)
    manifest = read_json(data_dir / "split_manifest.json")
    if not manifest or manifest.get("schema_version") != 1:
        raise ValueError("Missing supported frozen split manifest")
    hashes = manifest.get("artifact_sha256", {})
    required = ["train_tasks.jsonl", "validation_tasks.jsonl", "holdout_tasks.jsonl", "training_examples.jsonl"]
    if any(name not in hashes for name in required):
        raise ValueError("Split manifest must bind all task partitions and existing labels")
    for name, expected in hashes.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or file_sha(data_dir / relative) != expected:
            raise ValueError("Frozen data artifact changed: " + name)
    partitions, sources, assignments = {}, {}, manifest["source_assignment"]
    for split in ("train", "validation", "holdout"):
        rows = read_jsonl(data_dir / (split + "_tasks.jsonl"))
        keys, source_languages = set(), set()
        sources[split] = set()
        for row in rows:
            source = source_from_identity(row)
            if source != row.get("source_id") or assignments.get(source) != split:
                raise ValueError("Task identity disagrees with its source partition")
            key = task_id(row)
            identity = (source, row["language"])
            if key in keys or identity in source_languages:
                raise ValueError("Duplicate frozen task or source/language")
            keys.add(key); source_languages.add(identity); sources[split].add(source)
        if sources[split] != set(manifest["source_ids"][split]):
            raise ValueError("Source manifest does not match task partition")
        partitions[split] = rows
    if (sources["train"] & (sources["validation"] | sources["holdout"])
            or sources["validation"] & sources["holdout"]):
        raise ValueError("Cross-language source leakage between data partitions")
    if set(assignments) != set().union(*sources.values()):
        raise ValueError("Source assignment contains unaccounted identities")
    existing = read_jsonl(data_dir / "training_examples.jsonl")
    validate_training_examples(existing, [Task(**row) for row in partitions["train"]])
    return manifest, partitions, existing


def missing_tasks(data_dir):
    """Return only original frozen TRAIN identities without a verified label."""
    _, partitions, existing = _frozen_data(data_dir)
    covered = {(normalize_source_id(row["source_id"]), row["language"]) for row in existing}
    return [row for row in partitions["train"] if (row["source_id"], row["language"]) not in covered]


def _freeze_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temporary, path)  # Publish complete bytes without replacing an existing corpus.
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError("Immutable augmented corpus changed: " + str(path))
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def augment_training_corpus(data_dir, teacher_eval_dir, output_jsonl):
    data_dir, teacher_eval_dir, output_jsonl = map(Path, (data_dir, teacher_eval_dir, output_jsonl))
    manifest, partitions, existing = _frozen_data(data_dir)
    if output_jsonl.resolve() in {(data_dir / name).resolve() for name in manifest["artifact_sha256"]}:
        raise ValueError("Augmentation must not overwrite frozen data")
    train = {task_id(row): row for row in partitions["train"]}
    covered = {(row["source_id"], row["language"]) for row in existing}
    missing = {key: row for key, row in train.items() if (row["source_id"], row["language"]) not in covered}
    plan_path = teacher_eval_dir / "evaluation_plan.json"
    plan = read_json(plan_path)
    if not plan or plan.get("split") != "train_teacher":
        raise ValueError("Only explicit train_teacher evaluation can supply labels")
    if len(plan.get("candidates", [])) != 1:
        raise ValueError("Teacher stage requires exactly one frozen AR configuration")
    config = plan["candidates"][0]
    validate_candidate(config)
    expected_id = "ar-" + digest({k: v for k, v in config.items() if k != "id"})[:16]
    if ((config["model"], config["revision"]) != AR or config["decoder"] != "ar"
            or config.get("adapter") is not None or config.get("id") != expected_id):
        raise ValueError("Teacher must be the unchanged pinned AR model/configuration")
    if file_sha(plan["tasks_path"]) != plan.get("tasks_sha256"):
        raise ValueError("Teacher task file changed after plan freeze")
    expected = {}
    for task in read_jsonl(plan["tasks_path"]):
        key = task_id(task)
        if key in expected or key not in missing or digest(task) != digest(missing[key]):
            raise ValueError("Teacher task is not an exact missing TRAIN task")
        if source_from_identity(task) != task["source_id"]:
            raise ValueError("Teacher task source identity mismatch")
        expected[key] = task
    plan_hash, config_hash = file_sha(plan_path), digest(config)
    observed, additions, evidence, rejected = set(), [], {}, []
    for path in sorted(teacher_eval_dir.glob("shard-*/cases/*/*.json")):
        shard_dir = path.parents[2]
        binding = read_json(shard_dir / "binding.json")
        if not binding or binding.get("plan_sha256") != plan_hash:
            raise ValueError("Teacher shard has no matching frozen plan binding")
        row = read_json(path)
        key = row.get("task_key")
        task = expected.get(key)
        if key in observed or task is None:
            raise ValueError("Duplicate or unexpected teacher result")
        if (row.get("split") != "train_teacher" or row.get("task_sha256") != digest(task)
                or row.get("source_id") != task["source_id"] or row.get("language") != task["language"]
                or row.get("task_id") != task["task_id"] or source_from_identity(row) != task["source_id"]
                or row.get("candidate_id") != config["id"] or row.get("config_sha256") != config_hash
                or path.parent.name != config["id"] or path.stem != key):
            raise ValueError("Teacher result identity/configuration binding mismatch")
        if row.get("infrastructure_error") is not False or type(row.get("passed")) is not bool:
            raise ValueError("Teacher result lacks a completed trustworthy grading verdict")
        code = row.get("code")
        if not isinstance(code, str) or row.get("code_sha256") != digest(code):
            raise ValueError("Teacher implementation digest mismatch")
        observed.add(key)
        evidence[str(path.resolve())] = file_sha(path)
        if not row["passed"]:
            rejected.append({"task_key": key, "source_id": task["source_id"],
                             "language": task["language"], "status": row.get("status")})
            continue
        if row.get("status") != "passed" or not code.strip():
            raise ValueError("Passing teacher has no passing code result")
        additions.append({"example_id": task["source_id"] + "/" + task["language"],
            "source_id": task["source_id"], "language": task["language"],
            "messages": build_iterative_messages(task), "completion": fenced_completion(task["language"], code),
            "provenance": {"kind": "verified_ar_teacher", "synthetic": True,
                "model": AR[0], "revision": AR[1], "prompt_version": PROMPT_VERSION,
                "task_sha256": digest(task), "code_sha256": digest(code),
                "case_path": str(path.resolve()), "case_record_sha256": digest(row),
                "teacher_plan_sha256": plan_hash, "teacher_config_sha256": config_hash,
                "verified_on": "Unchanged TRAIN Task evaluator; no validation/holdout tests used"}})
    additions.sort(key=lambda row: (row["source_id"], row["language"]))
    combined = existing + additions
    validate_training_examples(combined, [Task(**row) for row in partitions["train"]])
    now_covered = {(row["source_id"], row["language"]) for row in combined}
    unlabeled = [{"task_key": key, "source_id": task["source_id"], "language": task["language"],
                  "reason": "teacher_failed" if key in observed else "not_evaluated"}
                 for key, task in sorted(missing.items())
                 if (task["source_id"], task["language"]) not in now_covered]
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in combined).encode()
    receipt = {"schema_version": 1, "kind": "train_only_verified_teacher_augmentation",
        "split_manifest_sha256": file_sha(data_dir / "split_manifest.json"),
        "original_training_sha256": file_sha(data_dir / "training_examples.jsonl"),
        "teacher_plan_sha256": plan_hash, "teacher_config_sha256": config_hash,
        "teacher_model": AR[0], "teacher_revision": AR[1], "teacher_evidence_sha256": evidence,
        "existing_labels_preserved": len(existing), "new_verified_synthetic_labels": len(additions),
        "training_examples": len(combined), "expected_teacher_cases": len(expected),
        "observed_teacher_cases": len(observed), "failed_teacher_cases": rejected,
        "unlabeled_count": len(unlabeled), "unlabeled": unlabeled,
        "validation_or_holdout_labels_used": 0, "output_sha256": hashlib.sha256(payload).hexdigest(),
        "output_jsonl": str(output_jsonl.resolve())}
    _freeze_bytes(output_jsonl, payload)
    freeze_json(output_jsonl.with_suffix(".receipt.json"), receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--teacher-eval-dir", type=Path)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--missing-only", action="store_true")
    args = parser.parse_args(argv)
    if args.missing_only:
        rows = missing_tasks(args.data_dir)
        _freeze_bytes(args.output_jsonl, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True)+"\n" for row in rows).encode())
        print(json.dumps({"missing_tasks": len(rows), "output_sha256": file_sha(args.output_jsonl)}))
    else:
        if args.teacher_eval_dir is None:
            parser.error("--teacher-eval-dir is required for augmentation")
        receipt = augment_training_corpus(args.data_dir, args.teacher_eval_dir, args.output_jsonl)
        print(json.dumps({key: receipt[key] for key in
            ("existing_labels_preserved", "new_verified_synthetic_labels", "training_examples", "unlabeled_count", "output_sha256")}))


if __name__ == "__main__":
    main()
