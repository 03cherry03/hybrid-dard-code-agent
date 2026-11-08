#!/usr/bin/env python3
"""One multilingual, matched AR/DARD/standard comparison within a fixed deadline.

Only source identities and calibration runtimes select the evaluation cohort.
Correctness results never choose tasks, decoder budgets, or the winning method.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import four_gpu
from scripts.all_experiments import (Interrupted, install_handlers, read_json,
                                     restore_handlers, write_json)
from scripts.deadline_budget import estimate_seconds
from scripts.focused_selection import build_selection, selection_manifest
from scripts.priority_experiments import calibration_records
from experiment_shards import select_shard
from datasets_code import load_tasks

BACKENDS = ["ar", "dard", "standard"]
SUITES = ["humaneval_plus", "multiple_rs", "multiple_js", "multiple_cpp",
          "mbpp", "multiple_mbpp_rs", "multiple_mbpp_js", "multiple_mbpp_cpp"]
ANALYSIS_RESERVE = 1800
MAX_EVALUATION_SOURCES = 360
DEFAULT_GENERATION_SETTINGS = {
    "tokens": 512, "standard_steps": 64, "block_size": 512, "dard_steps": 64,
    "tau_c": .5, "tau_u": .8, "transfer_limit": 0,
}


def generation_settings(args):
    """One explicit operating point shared by calibration and evaluation.

    Both diffusion methods use the full 512-token canvas by default. DARD uses
    one block rather than truncating its conditioning canvas into shorter blocks.
    Zero transfer_limit permits every threshold-eligible DARD position to be
    committed in parallel. It does not change the decoder's bounded fallback.
    """
    return {name: getattr(args, name, default)
            for name, default in DEFAULT_GENERATION_SETTINGS.items()}


def _append_record(path, record):
    """Small append-only local lifecycle records, durable before continuing."""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _stage(args, state, name, **details):
    state.update(stage=name, stage_started_unix=time.time(), **details)
    write_json(args.output / "focused_state.json", state)
    _append_record(args.output / "focused_events.jsonl", {
        "invocation_id": state["invocation_id"], "event": "stage_started",
        "stage": name, "timestamp_unix": time.time(), **details})


def _record_failure(args, state, *, error_type, error, exit_code, trace=None):
    record = {"schema_version": 1, "invocation_id": state["invocation_id"],
              "stage": state["stage"], "timestamp_unix": time.time(),
              "deadline_unix": args.deadline_unix, "error_type": error_type,
              "error": error, "exit_code": exit_code, "traceback": trace,
              "traceback_scope": "local_artifact_only_not_notification_body"}
    # The supervisor owns notifications. Local exceptions can contain paths or
    # model details, so only its sanitized failure summary should be mailed.
    write_json(args.output / "focused_failure.json", record)
    _append_record(args.output / "focused_failures.jsonl", record)


def freeze_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Frozen experiment selection/settings changed: " + str(path))
    else:
        write_json(path, value)


def campaign_argv(args, *, output, manifest, split="test"):
    settings = generation_settings(args)
    generation = [item for name, value in settings.items()
                  for item in ("--" + name.replace("_", "-"), str(value))]
    return ["functions", "--output", str(output), "--data-root", str(args.data_root),
            "--gpus", args.gpus, "--backends", ",".join(BACKENDS), "--suites", ",".join(SUITES),
            "--selection-manifest", str(manifest), "--repeats", "1", "--repairs", "0",
            "--seed", "42", *generation, "--quant", "bf16", "--diagnostics", "summary",
            "--split", split, "--execute", "--fail-fast"]


def candidate_workers(base_plan, tasks, manifest):
    # Match the coordinator's original task order and its exact shard algorithm.
    keys = {tuple(key) for key in manifest["task_keys"]}
    chosen = [task for task in tasks if tuple(four_gpu.task_key(task)) in keys]
    if len(chosen) != len(keys):
        raise ValueError("Prepared task identities differ from the selected cohort")
    workers = []
    for index, original in enumerate(base_plan["workers"]):
        selected = select_shard(chosen, 4, index)
        worker = copy.deepcopy(original)
        worker.update(task_keys=[four_gpu.task_key(task) for task in selected],
                      task_count=len(selected), expected_cases=3*len(selected),
                      task_seeds=[[42] for _ in selected],
                      by_dataset_language=dict(Counter(f"{t.dataset}/{t.language}" for t in selected)))
        workers.append(worker)
    return workers


def choose_cohort(catalog, tasks, base_plan, calibration_plan, shards, available_seconds):
    """Largest nested matched source cohort estimated to fit all three methods."""
    smallest = None
    for count in range(min(MAX_EVALUATION_SOURCES, len(catalog["evaluation_sources"])), 1, -1):
        manifest = selection_manifest(catalog, source_limit=count)
        workers = candidate_workers(base_plan, tasks, manifest)
        estimate = estimate_seconds(calibration_plan, shards, workers, BACKENDS, 1)
        choice = {"selected_source_count": count, "selected_task_count": manifest["task_count"],
                  "expected_cases": 3*manifest["task_count"], "estimate": estimate,
                  "fits_estimate": estimate["estimated_wall_seconds"] <= available_seconds,
                  "available_seconds": available_seconds}
        if choice["fits_estimate"]:
            return manifest, choice
        smallest = (manifest, choice)
    # Preserve every language, family and method even when no planned complete
    # comparison fits. The hard parent deadline still bounds this explicit risk.
    manifest, choice = smallest
    choice["deadline_at_risk"] = True
    return manifest, choice


def _run(args, state):
    root = args.output.parent
    settings = generation_settings(args)
    _stage(args, state, "freeze_generation_settings", generation_settings=settings)
    freeze_json(root / "generation_settings.json", settings)
    _stage(args, state, "load_prepared_tasks")
    tasks = load_tasks(SUITES, args.data_root / "datasets/function-cache", 0, 0, data_root=args.data_root)
    catalog = build_selection(tasks)
    _stage(args, state, "freeze_selection_catalog", catalog_sha256=catalog["catalog_sha256"])
    freeze_json(root / "selection_catalog.json", catalog)
    calibration_path = root / "calibration_selection.json"
    freeze_json(calibration_path, selection_manifest(catalog, split="calibration"))
    selection_path = root / "evaluation_selection.json"
    decision_path = args.output / "budget_decision.json"
    decision = read_json(decision_path)
    if decision:
        _stage(args, state, "validate_resume")
        if (decision["deadline_unix"] != args.deadline_unix or decision["backends"] != BACKENDS
                or decision["catalog_sha256"] != catalog["catalog_sha256"]
                or decision.get("generation_settings") != settings):
            raise ValueError("A resumed experiment must retain its original deadline and selection")
        selected = selection_manifest(catalog, source_limit=decision["selected_source_count"])
        if decision["selected_source_count"] > MAX_EVALUATION_SOURCES:
            raise ValueError("Resumed selection exceeds the fixed 1440-task evaluation scope")
        freeze_json(selection_path, selected)
    else:
        _stage(args, state, "calibration_deadline_check")
        if time.time() >= args.deadline_unix - ANALYSIS_RESERVE:
            raise TimeoutError("No time remains for calibration and a complete three-way comparison")
        calibration = root / "calibration"
        _stage(args, state, "calibration", campaign_output=str(calibration))
        code = four_gpu.main(campaign_argv(args, output=calibration, manifest=calibration_path, split="dev"))
        if code:
            return code
        _stage(args, state, "validate_calibration")
        calibration_plan, shards = calibration_records(calibration)
        # This full-size selection is a planning artifact only, not launched.
        planning_selection = root / "planning_full_selection.json"
        freeze_json(planning_selection, selection_manifest(catalog))
        _stage(args, state, "estimate_evaluation_cohort")
        target_args = four_gpu.parser().parse_args(campaign_argv(args, output=args.output, manifest=planning_selection))
        base_plan, _ = four_gpu.make_plan(target_args, four_gpu.resolve_gpus(args.gpus))
        available = max(0, args.deadline_unix-time.time()-ANALYSIS_RESERVE)
        selected, choice = choose_cohort(catalog, tasks, base_plan, calibration_plan, shards, available)
        freeze_json(selection_path, selected)
        decision = {**choice, "schema_version": 1, "status": "scheduled", "backends": BACKENDS,
                    "deadline_unix": args.deadline_unix, "decided_unix": time.time(),
                    "catalog_sha256": catalog["catalog_sha256"], "repeats": 1, "repairs": 0,
                    "all_languages_retained": ["py", "rs", "js", "cpp"],
                    "all_backends_required": True, "generation_budgets_unchanged": True,
                    "generation_settings": settings,
                    "comparison_scope": "One fixed parallel operating point; not a speed-quality Pareto sweep",
                    "analysis_reserve_seconds": ANALYSIS_RESERVE,
                    "selection_policy": "Nested common-source cohort chosen using calibration runtime, never correctness",
                    "maximum_evaluation_source_count": min(MAX_EVALUATION_SOURCES, len(catalog["evaluation_sources"])),
                    "selection_manifest_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest()}
        write_json(decision_path, decision)
    _stage(args, state, "evaluation_deadline_check", selected_source_count=decision["selected_source_count"],
           selected_task_count=selected["task_count"])
    if hashlib.sha256(selection_path.read_bytes()).hexdigest() != decision["selection_manifest_sha256"]:
        raise ValueError("Frozen evaluation selection file changed")
    if time.time() >= args.deadline_unix - 60:
        raise TimeoutError("Fixed experiment deadline reached")
    returned = False
    try:
        _stage(args, state, "evaluation", campaign_output=str(args.output))
        code = four_gpu.main(campaign_argv(args, output=args.output, manifest=selection_path))
        returned = True  # four_gpu.main already wrote exact coverage and paired analyses.
        decision.update(status="completed" if code == 0 else "failed", ended_unix=time.time())
        write_json(decision_path, decision)
        return code
    except BaseException as exc:
        decision.update(status="failed", error_type=type(exc).__name__, ended_unix=time.time())
        try:
            write_json(decision_path, decision)
        except OSError:
            pass
        raise
    finally:
        if not returned:
            try:
                plan = read_json(args.output / "campaign.json")
                if plan:
                    four_gpu.audit(plan, args.output)
                    from campaign_reporting import write_goal_report
                    write_goal_report(args.output)
            except Exception as exc:
                # Keep the primary execution error and preserve partial evidence.
                try:
                    write_json(args.output / "partial_report_error.json", {"error_type": type(exc).__name__})
                except OSError:
                    pass


def run(args):
    """Record every failure phase, including setup and calibration before selection."""
    args.output.mkdir(parents=True, exist_ok=True)
    state = {"schema_version": 1, "invocation_id": uuid.uuid4().hex,
             "status": "running", "stage": "start", "started_unix": time.time(),
             "deadline_unix": args.deadline_unix, "backends": list(BACKENDS),
             "repeats": 1, "repairs": 0, "maximum_evaluation_tasks": 4*MAX_EVALUATION_SOURCES}
    try:
        _stage(args, state, "start")
        code = _run(args, state)
        state.update(status="completed" if code == 0 else "failed", exit_code=code,
                     ended_unix=time.time())
        if code:
            _record_failure(args, state, error_type="CampaignExitFailure",
                            error=f"{state['stage']} returned nonzero exit code {code}", exit_code=code)
        write_json(args.output / "focused_state.json", state)
        _append_record(args.output / "focused_events.jsonl", {
            "invocation_id": state["invocation_id"], "event": state["status"],
            "stage": state["stage"], "timestamp_unix": state["ended_unix"], "exit_code": code})
        return code
    except BaseException as exc:
        interrupted = isinstance(exc, (KeyboardInterrupt, Interrupted))
        code = 124 if isinstance(exc, TimeoutError) else (130 if interrupted else 1)
        if isinstance(exc, Interrupted):
            code = 128 + exc.signum
        if isinstance(exc, SystemExit) and isinstance(exc.code, int):
            code = exc.code
        state.update(status="deadline_exceeded" if isinstance(exc, TimeoutError) else
                     ("interrupted" if interrupted else "failed"), exit_code=code,
                     error_type=type(exc).__name__, ended_unix=time.time())
        try:
            write_json(args.output / "focused_state.json", state)
            _record_failure(args, state, error_type=type(exc).__name__, error=str(exc),
                            exit_code=code, trace=traceback.format_exc())
        except OSError:
            pass  # Preserve the initiating failure if its filesystem is broken.
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("functions",))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("/data/hybrid-dard"))
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--deadline-unix", type=float)
    for name, default in DEFAULT_GENERATION_SETTINGS.items():
        p.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    actions = p.add_mutually_exclusive_group(required=True)
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--analyze", action="store_true")
    args = p.parse_args(argv)
    if args.analyze:
        from campaign_reporting import write_goal_report
        write_goal_report(args.output)
        report = read_json(args.output / "campaign_summary.json", {})
        groups = report.get("groups", {})
        complete = (report.get("coverage_complete") is True
                    and set(report.get("backends", [])) == set(BACKENDS)
                    and all(groups.get(f"{backend}/all", {}).get("complete_evaluated_cohort") is True
                            for backend in BACKENDS))
        return 0 if complete else 1
    if args.deadline_unix is None or not math.isfinite(args.deadline_unix) or args.deadline_unix <= 0:
        p.error("A fixed positive finite --deadline-unix is required")
    previous = install_handlers()
    try:
        return run(args)
    finally:
        restore_handlers(previous)


if __name__ == "__main__":
    raise SystemExit(main())
