#!/usr/bin/env python3
"""Paired AR / diffusion / DARD function-agent experiments, with no training."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback

from dard_decoder import DARDConfig
from datasets_code import CODE_GENERATION_SUITES, SUITES, load_tasks
from evaluation import strip_code, preflight_local, test_task, grade_final, category
from experiment_shards import SHARDING_ALGORITHM, select_manifest, select_shard, validate_shard
from reporting import case_key, latest_cases, summarize
from storage import data_root

ROOT = Path(__file__).resolve().parent
PROTOCOL = "paired_function_agent_local_sealed_grading_v6"
DEFAULT_SUITES = ",".join(CODE_GENERATION_SUITES)


class _FailFast(Exception):
    """Stop the campaign after persisting the first infrastructure failure."""


def cli(argv=None):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--suites", default=DEFAULT_SUITES)
    p.add_argument("--data-root", type=Path, default=data_root())
    p.add_argument("--allow-data-download", action="store_true", help="Legacy suites only; prefer prepare_data.py --download")
    p.add_argument("--list-suites", action="store_true")
    p.add_argument("--limit", type=int, default=20, help="Tasks per suite; 0 means all")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--selection-manifest", type=Path,
                   help="Exact public task identities before sharding; requires --limit 0 --start 0")
    p.add_argument("--selection-manifest-sha256", help=argparse.SUPPRESS)
    p.add_argument("--num-shards", type=int, default=1, help="Disjoint balanced partitions after suite start/limit selection")
    p.add_argument("--shard-index", type=int, default=0, help="Zero-based partition; keep its output and GPU assignment fixed")
    p.add_argument("--backends", "--decoders", dest="decoders", default="ar,standard,dard")
    p.add_argument("--model", default="apple/DiffuCoder-7B-Instruct")
    p.add_argument("--ar-model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--revision", default=None, help="Diffusion commit/tag; resolved commit recorded")
    p.add_argument("--ar-revision", default=None)
    p.add_argument("--local-files-only", action="store_true", help="Do not download model/tokenizer files")
    p.add_argument("--device", default="cuda:0", help="One logical CUDA device shared by all backends")
    p.add_argument("--quant", choices=("bf16", "4bit"), default="bf16")
    p.add_argument("--runner", choices=("none", "local"), default="local")
    p.add_argument("--repairs", type=int, default=1)
    p.add_argument("--tokens", type=int, default=256, help="Common maximum budget; diffusion uses fixed slots")
    p.add_argument("--standard-steps", type=int, default=64)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--dard-steps", type=int, default=64, help="Maximum steps per block")
    p.add_argument("--tau-c", type=float, default=.5)
    p.add_argument("--tau-u", type=float, default=.8)
    p.add_argument("--transfer-limit", type=int, default=0,
                   help="Maximum threshold-eligible DARD commits per step; 0 permits all")
    p.add_argument("--allow-attention-fallback", action="store_true", help="Diagnostic only; excludes controlled DARD speed claims")
    p.add_argument("--no-compact-queries", action="store_true",
                   help="Retain unused original predictor rows for numerical comparison")
    p.add_argument("--no-rectangular-attention", action="store_true",
                   help="Use full square attention for numerical comparison")
    p.add_argument("--diagnostics", choices=("summary", "steps", "profile"), default="summary",
                   help="Detailed steps/profile affect timing; use summary for speed comparisons")
    p.add_argument("--max-step-records", type=int, default=256, help="Maximum retained decoder step records per generation")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seeds", default=None, help="Explicit trial seeds; overrides --seed/--repeats")
    p.add_argument("--warmup", type=int, default=1, help="Unscored synthetic requests per backend load")
    p.add_argument("--test-timeout", type=int, default=45)
    p.add_argument("--grading-timeout", type=int, default=300)
    p.add_argument("--accuracy-tolerance", type=float, default=0.0, help="Descriptive tolerated accuracy loss, fraction")
    p.add_argument("--bootstrap-draws", type=int, default=1000)
    p.add_argument("--fail-fast", action="store_true",
                   help="Stop on infrastructure/model failures; incorrect generated code is a normal result")
    p.add_argument("--output", type=Path)
    p.add_argument("--check-data", action="store_true", help="Check prepared datasets and exit; offline by default")
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args(argv)
    a.suites = [x.strip() for x in a.suites.split(",") if x.strip()]
    a.decoders = [x.strip() for x in a.decoders.split(",") if x.strip()]
    if not a.suites or len(set(a.suites)) != len(a.suites) or any(x not in CODE_GENERATION_SUITES for x in a.suites):
        p.error("Select unique code-generation suites from --list-suites; library-use evaluation is excluded")
    if not a.decoders or len(set(a.decoders)) != len(a.decoders) or any(x not in ("ar", "standard", "dard") for x in a.decoders):
        p.error("--backends must be a unique subset of ar,standard,dard")
    if a.limit < 0 or a.start < 0 or a.repairs not in (0, 1, 2):
        p.error("--limit/--start >= 0; --repairs in {0,1,2}")
    if a.selection_manifest is not None and (a.limit or a.start):
        p.error("--selection-manifest defines the selection; use --limit 0 --start 0")
    if a.selection_manifest_sha256 is not None and a.selection_manifest is None:
        p.error("--selection-manifest-sha256 requires --selection-manifest")
    try:
        validate_shard(a.num_shards, a.shard_index)
    except ValueError as exc:
        p.error(str(exc))
    if a.runner == "none" and a.repairs != 0 and not (a.check_data or a.self_check or a.list_suites):
        p.error("--runner none requires --repairs 0")
    if not 16 <= a.tokens <= 1024 or a.standard_steps < 1:
        p.error("--tokens [16,1024]; --standard-steps >= 1")
    if a.repeats < 1 or a.warmup < 0 or a.test_timeout < 1 or a.grading_timeout < 1 or a.bootstrap_draws < 1:
        p.error("repeats/test-timeout/bootstrap-draws > 0; warmup >= 0")
    if not 0 <= a.accuracy_tolerance <= 1:
        p.error("--accuracy-tolerance must be in [0,1]")
    if not a.device.startswith("cuda:") or not a.device[5:].isdigit():
        p.error("--device must identify one CUDA device, e.g. cuda:0")
    try:
        a.trial_seeds = [int(x.strip()) for x in a.seeds.split(",")] if a.seeds else [a.seed + i for i in range(a.repeats)]
    except ValueError:
        p.error("--seeds must contain integers")
    if not a.trial_seeds or any(x < 0 or x >= 2**32 for x in a.trial_seeds):
        p.error("seeds must be in [0, 2**32)")
    _dard_config(a).validate()
    return a


def _dard_config(a):
    return DARDConfig(block_size=a.block_size, steps_per_block=a.dard_steps,
                      tau_c=a.tau_c, tau_u=a.tau_u, transfer_limit=a.transfer_limit,
                      temperature=a.temperature, top_p=a.top_p,
                      allow_attention_fallback=a.allow_attention_fallback,
                      compact_queries=not getattr(a, "no_compact_queries", False),
                      rectangular_attention=not getattr(a, "no_rectangular_attention", False),
                      diagnostics=a.diagnostics, max_step_records=a.max_step_records)


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2))
        _sync_stream(stream)
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _sync_stream(stream):
    stream.flush()
    try:
        descriptor = stream.fileno()
    except (AttributeError, io.UnsupportedOperation):
        return  # In-memory streams used by embedding callers have no disk.
    os.fsync(descriptor)


def _append(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    _sync_stream(stream)


def _status(output, stage, *, state="running", case=False, **details):
    """Constant-size crash breadcrumb, written before a potentially long step."""
    if output is None:
        return
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / ("current_case.json" if case else "worker_status.json"), {
        "schema_version": 1, "pid": os.getpid(), "stage": stage, "state": state,
        "updated_unix": time.time(), "updated_utc": datetime.now(timezone.utc).isoformat(), **details})


def _environment():
    packages = {}
    for package in ("torch", "transformers", "accelerate", "bitsandbytes", "safetensors", "huggingface_hub"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        hardware = r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        hardware = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "nvidia_gpus": hardware}


def _gpu_processes(gpu_uuid):
    """Sample external compute load around cases; never modify other processes."""
    if not gpu_uuid:
        return {"available": False, "other_pids": []}
    try:
        devices = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=10)
        if devices.returncode or gpu_uuid not in {line.strip() for line in devices.stdout.splitlines()}:
            return {"available": False, "other_pids": []}
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode:
            return {"available": False, "other_pids": []}
        pids = []
        for line in r.stdout.splitlines():
            values = [x.strip() for x in line.split(",")]
            if len(values) == 2 and values[0] == gpu_uuid and values[1].isdigit() and int(values[1]) != os.getpid():
                pids.append(int(values[1]))
        return {"available": True, "other_pids": sorted(set(pids))}
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False, "other_pids": []}


def _case_seed(task, trial_seed, attempt=0):
    value = f"{task.family or task.dataset}/{task.language}/{task.source_id or task.task_id}"
    offset = int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)
    return (trial_seed + offset + attempt * 100003) % (2**32)


def _sealed_category(verdict):
    if verdict.get("passed") is None:
        return verdict.get("status", "not_graded")
    evidence = verdict.get("sealed_diagnostics", verdict.get("feedback", ""))
    return category(verdict["status"], evidence if isinstance(evidence, str) else json.dumps(evidence))


def _run_case(backend, task, decoder, repeat, trial_seed, args, cfg, traces, errors, phase):
    attempts = []
    previous_code = feedback = ""
    started = time.perf_counter()
    completed, failure_kind = True, None
    output = getattr(args, "diagnostic_output", args.output)
    case_identity = {"phase_id": phase["phase_id"], "decoder": decoder, "dataset": task.dataset,
                     "language": task.language, "task_id": task.task_id, "repeat": repeat, "seed": trial_seed}
    current_stage = "generation"
    for attempt in range(args.repairs + 1):
        attempt_start = time.perf_counter()
        input_code = previous_code
        response = None
        record = {"decoder": decoder, "dataset": task.dataset, "language": task.language,
                  "task_id": task.task_id, "repeat": repeat, "seed": trial_seed,
                  "attempt": attempt + 1, "generation_seed": _case_seed(task, trial_seed, attempt)}
        try:
            current_stage = "generation"
            _status(output, current_stage, case=True, **case_identity, attempt=attempt + 1,
                    generation_seed=record["generation_seed"], generation_budget_tokens=args.tokens)
            response = backend.generate(task.language, task.prompt, decoder,
                seed=record["generation_seed"], num_new_tokens=args.tokens,
                steps=args.standard_steps, dard_config=cfg, temperature=args.temperature,
                top_p=args.top_p, old_code=previous_code, error=feedback)
            raw_output = response["text"]
            previous_code = strip_code(raw_output)
            record.update(raw_output=raw_output,
                          raw_output_sha256=hashlib.sha256(raw_output.encode()).hexdigest())
            if args.repairs:
                current_stage = "public_feedback"
                _status(output, current_stage, case=True, **case_identity, attempt=attempt + 1,
                        code_sha256=hashlib.sha256(previous_code.encode()).hexdigest())
            verdict = (test_task(task, previous_code, args.runner, args.test_timeout) if args.repairs else
                       {"passed": None, "status": "feedback_disabled", "feedback": "", "test_wall_s": 0})
            record.update(code=previous_code, generation={k: v for k, v in response.items() if k != "text"},
                          test=verdict, error_category=("feedback_disabled" if not args.repairs else
                                                        category(verdict["status"], verdict["feedback"])))
            if verdict["passed"] is None and args.runner != "none" and args.repairs:
                completed, failure_kind = False, "evaluation_infrastructure"
                _append(errors, {**case_identity, "attempt": attempt + 1, "stage": current_stage,
                                 "error_type": "EvaluationInfrastructureError", "verdict": verdict})
        except (KeyboardInterrupt, SystemExit) as exc:
            _append(errors, {**case_identity, "attempt": attempt + 1, "stage": current_stage,
                             "error_type": type(exc).__name__, "exception": repr(exc),
                             "traceback": traceback.format_exc(),
                             "generation_diagnostics": getattr(exc, "generation_diagnostics", {})})
            _status(output, current_stage, state="interrupted", case=True, **case_identity,
                    attempt=attempt + 1, error_type=type(exc).__name__)
            raise
        except Exception as exc:
            completed = False
            failure_kind = "generation_error" if response is None else "evaluation_infrastructure"
            generation = ({k: v for k, v in response.items() if k != "text"} if response is not None
                          else dict(getattr(exc, "generation_diagnostics", {}) or {}))
            generation.setdefault("wall_s", time.perf_counter() - attempt_start)
            if response is None:
                generation["measurement_complete"] = False
                generation.setdefault("diagnostics", {"complete": False, "status": "unavailable"})
            record.update(code=previous_code, generation=generation,
                          test={"passed": None, "status": failure_kind, "feedback": str(exc), "test_wall_s": 0},
                          error_category=failure_kind)
            _append(errors, {**{k: record[k] for k in ("decoder", "dataset", "task_id", "repeat", "seed", "attempt")},
                             "stage": current_stage, "error_type": type(exc).__name__,
                             "exception": repr(exc), "traceback": traceback.format_exc(),
                             "generation_diagnostics": generation})
        record.update(code_sha256=hashlib.sha256(previous_code.encode()).hexdigest(),
                      code_changed_from_previous=(previous_code != input_code) if attempt else None,
                      repair_diff=("".join(difflib.unified_diff(input_code.splitlines(keepends=True),
                          previous_code.splitlines(keepends=True), fromfile="previous", tofile="current")) if attempt else None))
        record["generation"].setdefault("diagnostics_mode", cfg.diagnostics)
        record["generation"].setdefault("instrumentation_affects_timing", cfg.diagnostics != "summary")
        record["generation"].setdefault("measurement_complete", True)
        record["attempt_wall_s"] = time.perf_counter() - attempt_start
        _append(traces, record)
        attempts.append(record)
        verdict = record["test"]
        print(f"   attempt={attempt + 1} status={verdict['status']} generation={record['generation']['wall_s']:.3f}s", flush=True)
        if verdict["passed"] is True or verdict["passed"] is None:
            break
        feedback = verdict["feedback"] or verdict["status"]
    # Freeze the submission before any final evaluator is called. Its verdicts
    # never affect repair prompts, stopping decisions or candidate selection.
    agent_wall_s = time.perf_counter() - started
    first, last = attempts[0], attempts[-1]
    ungraded = {"passed": None, "status": "not_graded", "feedback": "", "test_wall_s": 0, "sealed": True}
    first_grade, final_grade = dict(ungraded), dict(ungraded)
    grading_wall_s = 0.0
    if args.runner != "none":
        grade_started = time.perf_counter()
        try:
            if completed:
                current_stage = "sealed_final_grading"
                _status(output, current_stage, case=True, **case_identity,
                        code_sha256=last["code_sha256"], timeout_seconds=args.grading_timeout)
                final_grade = grade_final(task, last["code"], args.runner, args.grading_timeout)
            if first["test"]["status"] != "generation_error":
                if completed and first["code"] == last["code"]:
                    first_grade = final_grade
                else:
                    current_stage = "sealed_first_grading"
                    _status(output, current_stage, case=True, **case_identity,
                            code_sha256=first["code_sha256"], timeout_seconds=args.grading_timeout)
                    first_grade = grade_final(task, first["code"], args.runner, args.grading_timeout)
            if completed and (final_grade["passed"] is None or first_grade["passed"] is None):
                completed, failure_kind = False, "grading_infrastructure"
                _append(errors, {**case_identity, "stage": current_stage,
                                 "error_type": "GradingInfrastructureError",
                                 "grading": {"first": first_grade, "final": final_grade}})
        except (KeyboardInterrupt, SystemExit) as exc:
            _append(errors, {**case_identity, "stage": current_stage, "error_type": type(exc).__name__,
                             "exception": repr(exc), "traceback": traceback.format_exc()})
            _status(output, current_stage, state="interrupted", case=True, **case_identity,
                    error_type=type(exc).__name__)
            raise
        except Exception as exc:
            completed, failure_kind = False, "grading_infrastructure"
            _append(errors, {**case_identity, "stage": current_stage, "error_type": type(exc).__name__,
                             "exception": repr(exc), "traceback": traceback.format_exc()})
        grading_wall_s = time.perf_counter() - grade_started
    row = {"decoder": decoder, "model": args.ar_model if decoder == "ar" else args.model,
           "dataset": task.dataset, "family": task.family or task.dataset, "language": task.language,
           "final_evaluator": task.final_evaluator,
           "task_id": task.task_id, "source_id": task.source_id or task.task_id,
           "repeat": repeat, "seed": trial_seed, "attempts": len(attempts),
           "case_completed": completed, "failure_kind": failure_kind,
           "first_pass": first_grade["passed"], "final_pass": final_grade["passed"],
           "first_status": first_grade["status"], "final_status": final_grade["status"],
           "feedback_first_pass": first["test"]["passed"], "feedback_final_pass": last["test"]["passed"],
           "grading": {"first": first_grade, "final": final_grade}, "evaluation_protocol": "sealed_final_v1",
           "feedback_protocol": "public_repair_feedback" if args.repairs else "disabled_single_generation",
           "first_error_category": _sealed_category(first_grade), "last_error_category": _sealed_category(final_grade),
           "feedback_first_error_category": first["error_category"], "feedback_last_error_category": last["error_category"],
           "diagnostics_mode": cfg.diagnostics,
           "instrumentation_affects_timing": any(a["generation"]["instrumentation_affects_timing"] for a in attempts),
           "measurement_complete": all(a["generation"]["measurement_complete"] for a in attempts),
           "repair_code_changes": sum(a.get("code_changed_from_previous") is True for a in attempts),
           "first_generation_wall_s": first["generation"]["wall_s"],
           "first_attempt_wall_s": first["attempt_wall_s"],
           "generation_wall_s": sum(a["generation"]["wall_s"] for a in attempts),
           "model_wall_s": sum(a["generation"].get("model_wall_s", 0) for a in attempts),
           "test_wall_s": sum(a["test"]["test_wall_s"] for a in attempts) + final_grade.get("test_wall_s", 0),
           "agent_wall_s": agent_wall_s, "offline_grading_wall_s": grading_wall_s,
           "total_wall_s": agent_wall_s + final_grade.get("test_wall_s", 0),
           "messages_sha256": first["generation"].get("messages_sha256"),
           "task_sha256": _hash(asdict(task)), "phase_id": phase["phase_id"], "warmup_done": args.warmup > 0,
           "model_identity_verified": phase.get("model_identity_verified", False),
           "max_gpu_peak_gib": max(a["generation"].get("gpu_peak_gib", 0) for a in attempts)}
    for name in ("model_forward_passes", "forced_tokens", "promoted_tokens", "revoked_tokens", "attention_fallback_steps",
                 "demoted_tokens", "u_reverified_tokens", "forced_forward_passes",
                 "visible_output_tokens", "generated_tokens", "output_slots"):
        row[name] = sum(a["generation"].get(name, 0) for a in attempts)
    _status(output, current_stage, state="case_failed" if failure_kind else "awaiting_result_persist",
            case=True, **case_identity, failure_kind=failure_kind, pending_result=row)
    return row


def _summary(results, metadata, output):
    _status(output, "analysis", expected_cases=sum(metadata.get("expected_groups", {}).values()),
            recorded_cases=len(results))
    try:
        report = summarize(results, metadata, output)
    except BaseException as exc:
        with (output / "errors.jsonl").open("a", encoding="utf-8") as errors:
            _append(errors, {"stage": "analysis", "error_type": type(exc).__name__,
                             "exception": repr(exc), "traceback": traceback.format_exc()})
        _status(output, "analysis", state="failed", error_type=type(exc).__name__)
        raise
    _status(output, "analysis", state="completed", recorded_cases=len(results))
    return report


def main(argv=None):
    args = cli(argv)
    if args.list_suites:
        for name in CODE_GENERATION_SUITES:
            info = SUITES[name]
            print(f"{name}: {info['language']} / {info['family']}")
        return 0
    if args.self_check:
        import torch
        print(f"Python={sys.version.split()[0]} PyTorch={torch.__version__} CUDA={torch.cuda.is_available()}")
        return 0
    args.data_root = data_root(args.data_root)
    tasks = load_tasks(args.suites, args.data_root / "datasets/function-cache", args.limit, args.start,
                       data_root=args.data_root, allow_download=args.allow_data_download)
    tasks, selection = select_manifest(tasks, args.selection_manifest, args.selection_manifest_sha256)
    full_task_manifest = [asdict(t) for t in tasks]
    identities = [(t.dataset, t.language, t.task_id) for t in tasks]
    if len(identities) != len(set(identities)):
        raise RuntimeError("Duplicate task identities")
    if tasks or args.num_shards != 1:
        tasks = select_shard(tasks, args.num_shards, args.shard_index)
    sharding = {"algorithm": SHARDING_ALGORITHM, "num_shards": args.num_shards,
                "shard_index": args.shard_index, "full_selection_sha256": _hash(full_task_manifest),
                "full_selection_count": len(full_task_manifest)}
    if args.num_shards != 1:
        print(f"[shard] {args.shard_index}/{args.num_shards}: {len(tasks)}/{len(full_task_manifest)} tasks", flush=True)
    if args.check_data:
        return 0
    if not tasks:
        print("No tasks selected.")
        return 2
    output = args.output or args.data_root / "results/functions" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
    output.mkdir(parents=True, exist_ok=True)
    args.diagnostic_output = output
    languages = {t.language for t in tasks}
    local_environments = {}
    if args.runner == "local":
        _status(output, "evaluator_preflight", languages=sorted(languages))
        local_environments = preflight_local(languages, tasks=tasks, data_root=args.data_root)
    cfg = _dard_config(args)
    task_manifest = [asdict(t) for t in tasks]
    settings = {"protocol": PROTOCOL, "decoders": args.decoders, "model": args.model, "ar_model": args.ar_model,
                "revision": args.revision, "ar_revision": args.ar_revision, "quant": args.quant, "device": args.device,
                "runner": args.runner, "repairs": args.repairs, "tokens": args.tokens,
                "feedback_protocol": "public_repair_feedback" if args.repairs else "disabled_single_generation",
                "selection_manifest": selection,
                "standard_steps": args.standard_steps, "dard_settings": asdict(cfg),
                "temperature": args.temperature, "top_p": args.top_p, "trial_seeds": args.trial_seeds,
                "warmup": args.warmup, "test_timeout": args.test_timeout, "grading_timeout": args.grading_timeout,
                "data_root": str(args.data_root), "evaluation_protocol": "sealed_final_v1",
                "suites": args.suites, "limit_per_suite": args.limit, "start_per_suite": args.start,
                "num_shards": args.num_shards, "shard_index": args.shard_index, "sharding": sharding,
                "accuracy_tolerance": args.accuracy_tolerance, "bootstrap_draws": args.bootstrap_draws,
                "fail_fast": args.fail_fast,
                "task_manifest_sha256": _hash(task_manifest), "local_environments": local_environments, "environment": _environment(),
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob("*.py"))}}
    config_path = output / "config.json"
    expected = Counter(f"{d}/{t.dataset}/{t.language}" for t in tasks for d in args.decoders for _ in args.trial_seeds)
    if config_path.exists():
        metadata = json.loads(config_path.read_text(encoding="utf-8"))
        if metadata.get("settings") != settings:
            previous = metadata.get("settings", {})
            changed = sorted(k for k in settings if previous.get(k) != settings[k])
            raise RuntimeError("Resume configuration/environment/data/source mismatch: " + ", ".join(changed) + "; use a new --output")
        saved_tasks = output / "tasks.json"
        if not saved_tasks.exists() or _hash(json.loads(saved_tasks.read_text(encoding="utf-8"))) != settings["task_manifest_sha256"]:
            raise RuntimeError("Saved tasks.json does not match the experiment task manifest")
    else:
        metadata = {"settings": settings, "protocol": PROTOCOL, "expected_groups": dict(expected),
                    "suite_families": {t.dataset: t.family or t.dataset for t in tasks},
                    "accuracy_tolerance": args.accuracy_tolerance, "bootstrap_draws": args.bootstrap_draws,
                    "started_utc": datetime.now(timezone.utc).isoformat(), "phases": [], "backend_identities": {},
                    "dard_status": "independent_Dream_adapter_not_official_paper_reproduction",
                    "scope": "common_function_generation_test_repair_agent; not repository-level editing"}
        _write_json(output / "tasks.json", task_manifest)
        _write_json(config_path, metadata)
    results_path = output / "case_results.jsonl"
    results = latest_cases(results_path)
    done = {case_key(row) for row in results if row.get("case_completed", True)}
    def row_key(decoder, task, repeat, seed):
        return (decoder, task.dataset, task.language, task.task_id, repeat, seed)
    pending = [(r, seed, d, t) for r, seed in enumerate(args.trial_seeds) for d in args.decoders for t in tasks
               if row_key(d, t, r, seed) not in done]
    if not pending:
        _summary(results, metadata, output)
        print("[resume] All selected cases are complete.")
        return 0
    from models import create_backend
    interrupted = False
    invocation_failed = False
    with results_path.open("a", encoding="utf-8") as cases, (output / "runs.jsonl").open("a", encoding="utf-8") as traces, \
            (output / "errors.jsonl").open("a", encoding="utf-8") as errors:
        try:
            for repeat, seed in enumerate(args.trial_seeds):
                offset = repeat % len(args.decoders)
                order = args.decoders[offset:] + args.decoders[:offset]
                ordered_tasks = list(tasks)
                random.Random(seed).shuffle(ordered_tasks)
                for decoder in order:
                    todo = [t for t in ordered_tasks if row_key(decoder, t, repeat, seed) not in done]
                    if not todo:
                        continue
                    phase = {"phase_id": len(metadata["phases"]), "decoder": decoder, "repeat": repeat,
                             "seed": seed, "backend_order": order, "started_utc": datetime.now(timezone.utc).isoformat()}
                    metadata["phases"].append(phase)
                    _write_json(config_path, metadata)
                    backend = None
                    phase_stage = "model_load"
                    phase_start = time.perf_counter()
                    try:
                        print(f"[phase] repeat={repeat + 1} backend={decoder} cases={len(todo)} device={args.device}", flush=True)
                        _status(output, phase_stage, phase_id=phase["phase_id"], decoder=decoder, repeat=repeat,
                                model=args.ar_model if decoder == "ar" else args.model)
                        backend = create_backend(decoder, model_name=args.ar_model if decoder == "ar" else args.model,
                            quant=args.quant, device=args.device, revision=args.ar_revision if decoder == "ar" else args.revision,
                            local_files_only=args.local_files_only)
                        phase["model_load_s"] = time.perf_counter() - phase_start
                        details = backend.metadata()
                        identity_keys = ("model_name", "resolved_revision", "tokenizer_revision", "tokenizer_vocab_sha256",
                                         "model_config_sha256", "chat_template_sha256", "parameter_count")
                        identity = {key: details.get(key) for key in identity_keys}
                        identity_family = "ar" if decoder == "ar" else "diffusion"
                        previous = metadata["backend_identities"].get(identity_family)
                        if previous is not None and previous != identity:
                            raise RuntimeError("Model/tokenizer identity changed; use immutable checkpoint revisions")
                        metadata["backend_identities"][identity_family] = identity
                        phase["backend"] = details
                        phase["model_identity_verified"] = all(details.get(key) for key in
                            ("resolved_revision", "tokenizer_revision", "tokenizer_vocab_sha256",
                             "model_config_sha256", "chat_template_sha256"))
                        warmup_start = time.perf_counter()
                        for warmup in range(args.warmup):
                            phase_stage = "warmup"
                            _status(output, phase_stage, phase_id=phase["phase_id"], decoder=decoder,
                                    repeat=repeat, warmup=warmup + 1)
                            backend.generate("py", "Implement def add_one(x: int) -> int, returning x + 1.", decoder,
                                seed=seed + warmup, num_new_tokens=args.tokens, steps=args.standard_steps,
                                dard_config=cfg, temperature=args.temperature, top_p=args.top_p, old_code="", error="")
                        phase["warmup_s"] = time.perf_counter() - warmup_start
                        phase["warmup_requests"] = args.warmup
                        _write_json(config_path, metadata)
                        phase_stage = "cases"
                        _status(output, phase_stage, phase_id=phase["phase_id"], decoder=decoder, repeat=repeat)
                        for index, task in enumerate(todo, 1):
                            print(f"[{index}/{len(todo)}] {decoder} {task.dataset}/{task.task_id}", flush=True)
                            before = _gpu_processes(details.get("gpu_uuid"))
                            row = _run_case(backend, task, decoder, repeat, seed, args, cfg, traces, errors, phase)
                            after = _gpu_processes(details.get("gpu_uuid"))
                            row["resource_check_available"] = before["available"] and after["available"]
                            row["resource_contended"] = bool(before["other_pids"] or after["other_pids"])
                            row["other_gpu_pids"] = sorted(set(before["other_pids"] + after["other_pids"]))
                            _append(cases, row)
                            _status(output, "result_persisted", case=True,
                                    state="failed" if row["failure_kind"] else "completed",
                                    phase_id=phase["phase_id"], decoder=decoder, dataset=task.dataset,
                                    language=task.language, task_id=task.task_id, repeat=repeat, seed=seed,
                                    failure_kind=row["failure_kind"], case_completed=row["case_completed"])
                            if row["case_completed"]:
                                done.add(case_key(row))
                            if index % 5 == 0:
                                # Full reports reread the accumulated log and
                                # bootstrap every paired cohort. Repeating that
                                # every five cases stalls the GPU increasingly
                                # as the campaign grows. Cases are already
                                # durable; publish constant-size live progress
                                # here and summarize at phase/end boundaries.
                                _write_json(output / "progress.json", {
                                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                                    "phase_id": phase["phase_id"], "decoder": decoder,
                                    "repeat": repeat, "phase_cases_processed": index,
                                    "phase_cases_selected": len(todo),
                                    "completed_cases": len(done),
                                    "expected_cases": sum(expected.values()),
                                    "latest_task_id": task.task_id,
                                    "summary_refresh": "backend_phase_or_run_end",
                                })
                            if row["failure_kind"] == "generation_error":
                                raise RuntimeError("Generation failed; remaining cases in this phase are retryable")
                            if args.fail_fast and row["failure_kind"]:
                                raise RuntimeError("Infrastructure failed: " + row["failure_kind"])
                        phase["completed"] = True
                    except (KeyboardInterrupt, SystemExit) as exc:
                        phase["completed"] = False
                        phase["error"] = repr(exc)
                        phase["error_stage"] = phase_stage
                        _append(errors, {"phase_id": phase["phase_id"], "decoder": decoder, "repeat": repeat,
                                         "stage": phase_stage, "error_type": type(exc).__name__,
                                         "exception": repr(exc), "traceback": traceback.format_exc(),
                                         "generation_diagnostics": getattr(exc, "generation_diagnostics", {})})
                        raise
                    except Exception as exc:
                        phase["completed"] = False
                        phase["error"] = repr(exc)
                        phase["error_stage"] = phase_stage
                        _append(errors, {"phase_id": phase["phase_id"], "decoder": decoder, "repeat": repeat,
                                         "stage": phase_stage, "error_type": type(exc).__name__,
                                         "exception": repr(exc), "traceback": traceback.format_exc(),
                                         "generation_diagnostics": getattr(exc, "generation_diagnostics", {})})
                        print(f"[phase error] {decoder}: {exc}", flush=True)
                        if args.fail_fast:
                            raise _FailFast(str(exc)) from exc
                    finally:
                        cleanup_error = None
                        try:
                            if backend is not None:
                                _status(output, "model_cleanup", phase_id=phase["phase_id"],
                                        decoder=decoder, repeat=repeat)
                                backend.close()
                        except Exception as exc:
                            cleanup_error = exc
                            phase.update(completed=False, cleanup_error=repr(exc))
                            _append(errors, {"phase_id": phase["phase_id"], "decoder": decoder, "repeat": repeat,
                                             "stage": "model_cleanup", "error_type": type(exc).__name__,
                                             "exception": repr(exc), "traceback": traceback.format_exc()})
                        phase["phase_wall_s"] = time.perf_counter() - phase_start
                        _write_json(config_path, metadata)
                        _summary(latest_cases(results_path), metadata, output)
                        if cleanup_error is not None:
                            raise _FailFast("Backend cleanup failed") from cleanup_error
        except _FailFast:
            invocation_failed = True
            print("Stopped after infrastructure failure. Completed cases saved; unfinished cases remain retryable.", flush=True)
        except KeyboardInterrupt:
            interrupted = True
            print("Interrupted. Completed cases saved; unfinished cases remain retryable.", flush=True)
        finally:
            _summary(latest_cases(results_path), metadata, output)
    results = latest_cases(results_path)
    completed_keys = {case_key(row) for row in results if row.get("case_completed", True)}
    remaining = sum(row_key(d, t, r, seed) not in completed_keys for r, seed in enumerate(args.trial_seeds)
                    for d in args.decoders for t in tasks)
    print(f"[saved] {output.resolve()} ; unfinished cases={remaining}", flush=True)
    code = 130 if interrupted else (1 if remaining or invocation_failed else 0)
    _status(output, "run_end", state="completed" if code == 0 else "failed", exit_code=code,
            remaining_cases=remaining, completed_cases=len(completed_keys))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
