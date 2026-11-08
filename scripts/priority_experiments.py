#!/usr/bin/env python3
"""Run complete AR/DARD coverage first, then optional standard within a fixed deadline."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import four_gpu
from scripts.all_experiments import read_json, write_json
from scripts.deadline_budget import select_repeats
from campaign_reporting import write_priority_index

ANALYSIS_RESERVE = 1800


def campaign_argv(args, *, output, backends, repeats=1, limit=0):
    return ["functions", "--output", str(output), "--data-root", str(args.data_root),
            "--gpus", args.gpus, "--backends", ",".join(backends), "--limit", str(limit),
            "--repeats", str(repeats), "--seed", "42", "--tokens", "512", "--standard-steps", "512",
            "--block-size", "128", "--dard-steps", "192", "--quant", "bf16",
            "--diagnostics", "summary", "--split", "dev" if limit else "test",
            "--execute", "--fail-fast"]


def calibration_records(directory):
    plan = read_json(directory / "campaign.json")
    if not plan:
        raise ValueError("Missing full-budget calibration plan")
    status = read_json(directory / "execution_status.json", {})
    if status.get("exit_codes") != {str(w["index"]): 0 for w in plan["workers"]}:
        raise ValueError("Calibration did not complete successfully on all GPUs")
    shards = []
    for worker in plan["workers"]:
        path = Path(worker["output"])
        rows = [json.loads(line) for line in (path / "case_results.jsonl").read_text().splitlines() if line.strip()]
        shards.append({"rows": rows, "phases": read_json(path / "config.json", {})["phases"]})
    return plan, shards


def save_decision(args, decision):
    write_json(args.output / "budget_decision.json", decision)
    root = args.output.parent
    combined = read_json(root / "budget_decision.json", {})
    combined.update(deadline_unix=args.deadline_unix,
                    priority_order=["ar", "dard", "standard"], analysis_reserve_seconds=ANALYSIS_RESERVE)
    combined[args.mode] = decision
    write_json(root / "budget_decision.json", combined)
    write_priority_index(root)


def run(args):
    backends = ["ar", "dard"] if args.mode == "priority" else ["standard"]
    args.output.mkdir(parents=True, exist_ok=True)
    saved = read_json(args.output / "budget_decision.json")
    if args.mode == "standard" and not (saved and saved.get("status") == "skipped"):
        priority = write_priority_index(args.output.parent)
        if not priority["required_campaigns_complete"]:
            raise ValueError("Standard diffusion cannot start before complete evaluated AR/DARD coverage")
    if saved:
        if saved["deadline_unix"] != args.deadline_unix or saved["backends"] != backends:
            raise ValueError("Persisted deadline/backend decision changed; use a new output directory")
        if saved["status"] == "skipped":
            return 0
        repeats = saved["selected_repeats"]
        decision = saved
    else:
        calibration = args.calibration
        if args.mode == "standard":
            # Standard calibration is itself optional and starts only after the
            # priority result is durable. Never spend the final analysis reserve.
            if args.deadline_unix - time.time() <= ANALYSIS_RESERVE + 3600:
                decision = {"status": "skipped", "backends": backends, "selected_repeats": 0,
                            "deadline_unix": args.deadline_unix,
                            "reason": "Insufficient remaining time for standard calibration and full evaluation"}
                save_decision(args, decision)
                return 0
            calibration = args.output.parent / "standard-calibration"
            code = four_gpu.main(campaign_argv(args, output=calibration, backends=backends, limit=2))
            if code:
                return code
        calibration_plan, shards = calibration_records(calibration)
        target_args = four_gpu.parser().parse_args(campaign_argv(args, output=args.output, backends=backends))
        target_plan, _ = four_gpu.make_plan(target_args, four_gpu.resolve_gpus(args.gpus))
        available = max(0, args.deadline_unix - time.time() - ANALYSIS_RESERVE)
        choice = select_repeats(calibration_plan, shards, target_plan["workers"], backends,
                                available, max_repeats=3)
        repeats = choice["selected_repeats"]
        decision = {**choice, "status": "scheduled", "backends": backends,
                    "deadline_unix": args.deadline_unix, "decided_unix": time.time(),
                    "calibration": str(calibration), "all_tasks_retained": True,
                    "generation_budgets_unchanged": True}
        if repeats == 0:
            if args.mode == "standard":
                decision["status"] = "skipped"
                save_decision(args, decision)
                return 0
            # Required work is attempted without dropping tasks or weakening
            # decoding budgets. This is explicitly at risk, not a success claim.
            repeats = 1
            decision.update(selected_repeats=1, deadline_at_risk=True)
        save_decision(args, decision)
    if time.time() >= args.deadline_unix - 60:
        raise TimeoutError("Fixed experiment deadline reached")
    campaign_returned = False
    primary_error = None
    try:
        code = four_gpu.main(campaign_argv(args, output=args.output, backends=backends, repeats=repeats))
        # four_gpu.main audits before every normal execution return, including
        # a nonzero worker result. Repeating it reruns all paired bootstraps.
        campaign_returned = True
        decision.update(status="completed" if code == 0 else "failed", ended_unix=time.time())
        save_decision(args, decision)
        return code
    except BaseException as exc:
        primary_error = exc
        decision.update(status="failed", ended_unix=time.time(), error=repr(exc))
        try:
            save_decision(args, decision)
        except Exception:
            pass  # Preserve the original execution failure if reporting fails.
        raise
    finally:
        # Required results remain directly available even if later optional work
        # is interrupted. Partial cohorts retain their unknown/missing status.
        try:
            plan = read_json(args.output / "campaign.json")
            if plan and not campaign_returned:
                four_gpu.audit(plan, args.output)
            write_priority_index(args.output.parent)
        except Exception as exc:
            if primary_error is None:
                raise
            # A best-effort partial report must never replace the reason the
            # experiment failed (including interruption and deadline errors).
            decision["partial_analysis_error"] = repr(exc)
            try:
                write_json(args.output / "budget_decision.json", decision)
            except Exception:
                pass


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("functions",))
    p.add_argument("--mode", choices=("priority", "standard"), default="priority")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("/data/hybrid-dard"))
    p.add_argument("--calibration", type=Path)
    p.add_argument("--deadline-unix", type=float)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--analyze", action="store_true")
    args = p.parse_args(argv)
    if args.analyze:
        result = write_priority_index(args.output)
        return 0 if result["required_campaigns_complete"] else 1
    if (args.deadline_unix is None or not math.isfinite(args.deadline_unix)
            or args.deadline_unix <= 0 or args.calibration is None):
        p.error("A fixed --deadline-unix and --calibration are required")
    if not args.execute:
        p.error("Use --execute only after freezing the parent pipeline plan")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
