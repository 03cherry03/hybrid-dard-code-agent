#!/usr/bin/env python3
"""Reproduce the frozen CE-only r32/1200 validation and holdout report.

This command deliberately has no candidate search.  It trains the exact
masked-CE adapter (unless an existing adapter is supplied), then evaluates the
fixed AR/native/DARD trio with the report's seeds and grading protocol.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ce_only_reproduction import (EVALUATION, EXPERIMENT_ID, REPORT_TRAINING_SOURCE_SHA256, TRAINING,
    balance_training, reproduction_candidates,
    report_contract_manifest, result_contract, verify_report_artifacts)
from iterative_tuning import adapter_identity, file_sha, freeze_json, read_json, write_json
from scripts.iterative_experiments import evaluate, verify_plan
from scripts.run_focused_model_target import cuda_preflight, runtime_env

TRAINING_SOURCE_FILES = (
    Path("diffusion_training.py"),
    Path("iteration_observability.py"),
    Path("tokenizer_contract.py"),
    Path("scripts/train_diffusion_lora.py"),
)


def prepare_training_source(output: Path, source_root: Path = ROOT) -> Path:
    """Freeze the exact trainer implementation used by this output directory."""
    destination = output / "training_source"
    manifest_path = output / "training_source_manifest.json"
    if destination.exists() and manifest_path.is_file():
        manifest = read_json(manifest_path, {})
        expected_files = {str(relative) for relative in TRAINING_SOURCE_FILES}
        recorded_files = set(manifest.get("files", {}))
        if recorded_files != expected_files:
            raise ValueError("Training source manifest file set changed")
        for relative, expected in manifest["files"].items():
            actual = file_sha(destination / relative)
            if actual != expected:
                raise ValueError("Frozen training source changed: " + relative)
        return destination
    if destination.exists():
        raise ValueError("Training source exists without an immutable manifest: " + str(destination))
    destination.mkdir(parents=True)
    files = {}
    for relative in TRAINING_SOURCE_FILES:
        source = Path(source_root) / relative
        if not source.is_file():
            raise FileNotFoundError("Training source file is missing: " + str(source))
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[str(relative)] = file_sha(target)
    write_json(manifest_path, {"schema_version": 1, "source_root": str(Path(source_root).resolve()),
                               "files": files,
                               "report_files": REPORT_TRAINING_SOURCE_SHA256,
                               "matches_report_files": files == REPORT_TRAINING_SOURCE_SHA256})
    return destination


def training_command(args, train_jsonl: Path, training_output: Path, source_root: Path = ROOT):
    """Build the frozen four-rank training command.

    ``source_root`` defaults to this checkout for unit tests and local
    inspection.  A real reproduction passes the output's immutable training
    snapshot so all four ranks execute the same trainer implementation.
    """
    python = Path(args.python)
    return [str(python), "-m", "torch.distributed.run", "--standalone",
            "--nnodes=1", "--nproc_per_node=4",
            str(Path(source_root) / "scripts" / "train_diffusion_lora.py"),
            "--train-jsonl", str(train_jsonl), "--output", str(training_output),
            "--base-model", TRAINING["base_model"], "--revision", TRAINING["revision"],
            "--max-length", str(TRAINING["max_length"]), "--epochs", "0",
            "--max-steps", str(TRAINING["max_steps"]),
            "--batch-size", str(TRAINING["batch_size"]),
            "--gradient-accumulation", str(TRAINING["gradient_accumulation"]),
            "--learning-rate", str(TRAINING["learning_rate"]),
            "--weight-decay", str(TRAINING["weight_decay"]),
            "--warmup-steps", str(TRAINING["warmup_steps"]),
            "--max-grad-norm", str(TRAINING["max_grad_norm"]),
            "--rank", str(TRAINING["lora_rank"]),
            "--alpha", str(TRAINING["lora_alpha"]),
            "--target-modules", ",".join(TRAINING["target_modules"]),
            "--lora-dropout", str(TRAINING["lora_dropout"]),
            "--min-mask-prob", str(TRAINING["min_mask_prob"]),
            "--checkpoint-every", str(TRAINING["checkpoint_every"]),
            "--balanced-languages", "--seed", str(TRAINING["seed"]),
            "--deadline-unix", str(args.deadline_unix),
            "--deadline-reserve-seconds", str(args.deadline_reserve_seconds),
            "--local-files-only"]


def train(args, env, balanced_jsonl: Path, output: Path, source_root: Path = ROOT) -> Path:
    destination = output / "training"
    state_path = destination / "training_state.json"
    adapter = destination / "final_adapter"
    if state_path.is_file() and adapter.is_dir():
        state = read_json(state_path, {})
        if state.get("status") == "completed":
            config = read_json(destination / "training_config.json", {})
            expected = {
                "base_model": TRAINING["base_model"], "base_revision": TRAINING["revision"],
                "max_length": TRAINING["max_length"], "max_steps": TRAINING["max_steps"],
                "batch_size_per_rank": TRAINING["batch_size"],
                "gradient_accumulation": TRAINING["gradient_accumulation"],
                "learning_rate": TRAINING["learning_rate"], "weight_decay": TRAINING["weight_decay"],
                "warmup_steps": TRAINING["warmup_steps"], "max_grad_norm": TRAINING["max_grad_norm"],
                "lora_rank": TRAINING["lora_rank"], "lora_alpha": TRAINING["lora_alpha"],
                "lora_dropout": TRAINING["lora_dropout"], "min_mask_prob": TRAINING["min_mask_prob"],
                "seed": TRAINING["seed"], "balanced_languages": TRAINING["balanced_languages"],
                "world_size": TRAINING["world_size"],
            }
            for key, value in expected.items():
                if config.get(key) != value:
                    raise ValueError(f"Completed adapter training config changed at {key}")
            if tuple(config.get("lora_targets", ())) != TRAINING["target_modules"]:
                raise ValueError("Completed adapter LoRA target modules changed")
            return adapter
    destination.mkdir(parents=True, exist_ok=True)
    command = training_command(args, balanced_jsonl, destination, source_root=source_root)
    train_env = dict(env)
    old_pythonpath = train_env.get("PYTHONPATH")
    train_env["PYTHONPATH"] = str(Path(source_root).resolve()) + (
        os.pathsep + old_pythonpath if old_pythonpath else "")
    with (output / "training.stdout.log").open("a") as log:
        log.write(json.dumps({"event": "training_launch", "command": command,
                              "time": time.time()}, ensure_ascii=False) + "\n")
        log.flush()
        result = subprocess.run(command, cwd=str(source_root), env=train_env,
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed ({result.returncode}); see {output/'training.stdout.log'}")
    state = read_json(state_path, {})
    if state.get("status") != "completed" or not adapter.is_dir():
        raise RuntimeError("Training did not produce a completed adapter: " + json.dumps(state))
    return adapter


def frozen_plan(base_plan_path: Path, output: Path, args):
    plan = copy.deepcopy(read_json(base_plan_path))
    if not plan:
        raise ValueError("Base plan is empty: " + str(base_plan_path))
    # The base plan owns the audited source snapshot, model receipts and GPU
    # identities.  Only output, seed and the explicitly frozen protocol change.
    plan["output"] = str(output.resolve())
    plan["seed"] = EVALUATION["validation_seed"]
    plan["deadline_unix"] = args.deadline_unix
    plan["grading_timeout"] = EVALUATION["grading_timeout"]
    plan["generation_timeout"] = EVALUATION["generation_timeout"]
    plan["evaluation_options"] = {
        "extraction_version": EVALUATION["extraction_version"],
        "counterbalanced_order": EVALUATION["counterbalanced_order"],
        "candidate_warmup": EVALUATION["candidate_warmup"],
        "save_grading_verdict": True,
        "grading_workers": EVALUATION["grading_workers"],
        "grading_queue_size": EVALUATION["grading_queue_size"],
    }
    # Keep the frozen plan auditable and prevent a stale target from being
    # mistaken for this experiment.
    plan["campaign"] = EXPERIMENT_ID
    plan["source_plan"] = {"path": str(base_plan_path.resolve()),
                            "sha256": file_sha(base_plan_path)}
    return plan


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base_plan_path = args.base_plan.resolve()
    plan = frozen_plan(base_plan_path, output, args)
    verify_plan(plan)
    freeze_json(output / "iteration_plan.json", plan)
    data = Path(args.data or plan["data"]).resolve()
    train_jsonl = Path(args.train_jsonl or data / "training_examples.jsonl").resolve()
    validation_tasks = data / "validation_tasks.jsonl"
    holdout_tasks = data / "holdout_tasks.jsonl"
    if not train_jsonl.is_file() or not validation_tasks.is_file() or not holdout_tasks.is_file():
        raise FileNotFoundError("Training, validation, and holdout JSONL files are required")
    expected_data_hashes = plan.get("data_sha256", {})
    for name, path in (("training_examples.jsonl", train_jsonl),
                       ("validation_tasks.jsonl", validation_tasks),
                       ("holdout_tasks.jsonl", holdout_tasks)):
        expected = expected_data_hashes.get(name)
        if expected and file_sha(path) != expected:
            raise ValueError(f"Frozen plan data changed for {name}: {path}")
    balanced_jsonl = output / "data" / "balanced_training.jsonl"
    balance_receipt = balance_training(train_jsonl, balanced_jsonl)
    write_json(output / "data" / "balanced_training.receipt.json", balance_receipt)
    training_source = prepare_training_source(output)
    adapter_path = args.adapter_path.resolve() if args.adapter_path else None
    env = runtime_env()
    cuda_preflight(Path(args.python), env, output)
    if adapter_path is None:
        adapter_path = train(args, env, balanced_jsonl, output, source_root=training_source)
    adapter = adapter_identity(adapter_path)
    configs = reproduction_candidates(adapter_path)
    manifest = report_contract_manifest(output=output, base_plan=base_plan_path,
        train_jsonl=train_jsonl, balanced_jsonl=balanced_jsonl,
        validation_tasks=validation_tasks, holdout_tasks=holdout_tasks, adapter=adapter)
    manifest["training_source"] = read_json(output / "training_source_manifest.json", {})
    manifest["candidates"] = configs
    manifest["plan_sha256"] = file_sha(output / "iteration_plan.json") if (output / "iteration_plan.json").is_file() else None
    freeze_json(output / "reproduction_manifest.json", manifest)
    # evaluate() freezes each evaluation_plan.json and resumes completed rows.
    validation = evaluate(plan, "validation", configs, split="validation",
                          deadline=args.deadline_unix - 360)
    repeated = []
    for seed in EVALUATION["repeat_seeds"]:
        repeated_plan = copy.deepcopy(plan)
        repeated_plan["seed"] = seed
        repeated.append({"seed": seed, "rows": evaluate(
            repeated_plan, f"final_{seed}", configs, split="validation",
            deadline=args.deadline_unix - 240)})
    holdout_plan = copy.deepcopy(plan)
    holdout_plan["seed"] = EVALUATION["holdout_seed"]
    holdout = evaluate(holdout_plan, "holdout", configs, split="holdout",
                       deadline=args.deadline_unix - 120)
    result = {"status": "completed", "experiment_id": EXPERIMENT_ID,
              "adapter": adapter, "training": TRAINING,
              "validation": validation, "final": repeated, "holdout": holdout,
              "selected": configs[-1], "holdout_used_for_selection": False,
              "generated_unix": time.time(), "manifest": str(output / "reproduction_manifest.json")}
    write_json(output / "focused_full_result.json", result)
    write_json(output / "ce_only_summary.json", {"status": "completed",
        "objective": "masked_ce_only", "metrics": result_contract(result),
        "selected_dard_parameters": configs[-1]["dard"],
        "source_result": str(output / "focused_full_result.json"),
        "training": {k: TRAINING[k] for k in ("lora_rank", "max_steps", "learning_rate", "balanced_languages")}})
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-plan", type=Path,
                   help="Frozen plan with audited source/data/model/GPU identities")
    p.add_argument("--output", type=Path)
    p.add_argument("--verify-report", type=Path,
                   help="Audit an existing report directory without CUDA or model loading")
    p.add_argument("--verify-bundle", type=Path,
                   help="Verify the portable adapter/data bundle without CUDA or model loading")
    p.add_argument("--data", type=Path)
    p.add_argument("--train-jsonl", type=Path)
    p.add_argument("--adapter-path", type=Path,
                   help="Reuse a completed adapter and skip the 1,200-step training")
    p.add_argument("--python", type=Path, default=Path("/data/hybrid-dard/envs/iterative-training/bin/python"))
    p.add_argument("--deadline-unix", type=float)
    p.add_argument("--deadline-reserve-seconds", type=float, default=900)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.verify_report:
        print(json.dumps(verify_report_artifacts(args.verify_report), ensure_ascii=False, indent=2))
        raise SystemExit(0)
    if args.verify_bundle:
        from ce_only_reproduction import verify_bundle
        print(json.dumps(verify_bundle(args.verify_bundle), ensure_ascii=False, indent=2))
        raise SystemExit(0)
    if not args.base_plan or not args.output or args.deadline_unix is None:
        raise SystemExit("--base-plan, --output, and --deadline-unix are required unless --verify-report is used")
    raise SystemExit((run(args) and 0))
