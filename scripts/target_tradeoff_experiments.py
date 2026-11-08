#!/usr/bin/env python3
"""Bounded validation search for the user's three-way speed/accuracy target.

Reuse the audited inference and grading path; do not alter the decoder, old
results, or AR settings to manufacture the requested ordering.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iterative_tuning import (AR, candidate, digest, file_sha, freeze_json,
                             read_json, read_jsonl, validate_candidate, write_json)
from scripts.tradeoff_experiments import configured, shared_context
from scripts.iterative_experiments import evaluate, source_files, update, verify_plan
from scripts.four_gpu import owned_descendant_cleanup


def target_grid(previous_selection):
    reference = copy.deepcopy(next(r["candidate"] for r in previous_selection["winners_by_decoder"]
                                   if r["candidate"]["decoder"] == "ar"))
    configs = {reference["id"]: reference}
    def add(**kwargs):
        value = configured(**kwargs)
        configs[value["id"]] = value
        return value
    control = candidate(AR, decoder="ar", tokens=256, native_steps=1, temperature=0,
                        top_p=1, forward_budget=512, prefix="none")
    configs[control["id"]] = control
    for steps in (2, 4, 8, 16):
        add(decoder="standard", steps=steps)
    for temperature in (.1, .3):
        add(decoder="standard", steps=8, temperature=temperature, top_p=1.)
    add(decoder="standard", steps=128)
    add(decoder="standard", steps=256, temperature=.3, top_p=.95)
    add(steps=64, block=128, tau_c=.6, tau_u=.8)
    for tau_c in (.5, .6):
        for tau_u in (.9, .95):
            add(steps=64, block=256, tau_c=tau_c, tau_u=tau_u)
    for temperature in (.1, .3):
        for tau_u in (.9, .95):
            add(steps=64, block=256, tau_c=.6, tau_u=tau_u,
                temperature=temperature, top_p=1.)
        add(steps=128, block=256, tau_c=.6, tau_u=.9,
            temperature=temperature, top_p=1.)
    add(steps=128, block=256, tau_c=.6, tau_u=.9, temperature=.1,
        top_p=1., completion_policy="force_then_verify")
    # Throttle simultaneous proposals without changing the algorithm.
    for temperature in (0., .1):
        value = copy.deepcopy(configured(steps=128, block=256, tau_c=.6,
                              tau_u=.9, temperature=temperature, top_p=1.))
        value["dard"]["transfer_limit"] = 8
        value.pop("id")
        value["id"] = "dard-" + digest(value)[:16]
        validate_candidate(value)
        configs[value["id"]] = value
    return list(configs.values()), reference["id"]


def prepare(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "iteration_plan.json"
    if plan_path.exists():
        plan = read_json(plan_path)
        if plan["target"]["max_accuracy_gap"] != args.max_accuracy_gap:
            raise ValueError("Existing frozen target differs; use a new output directory")
        verify_plan(plan)
        return plan
    previous = args.previous.resolve()
    old = read_json(previous / "iteration_plan.json")
    verify_plan(old)
    configs, ar_id = target_grid(read_json(previous / "final_selection.json"))
    snapshot = output / "source"
    hashes = {}
    for path in source_files(ROOT):
        relative = path.relative_to(ROOT)
        dest = snapshot / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        hashes[str(relative)] = file_sha(dest)
    plan = {k: copy.deepcopy(old[k]) for k in
            ("data_root", "data", "data_sha256", "python", "gpus", "package_versions",
             "model_receipts", "grading_timeout", "generation_timeout", "evaluation_options")}
    plan.update(schema_version=1, campaign="three_way_target_feasibility_v1",
        output=str(output), source=str(snapshot), source_sha256=hashes,
        previous_plan={"path":str(previous/"iteration_plan.json"), "sha256":file_sha(previous/"iteration_plan.json")},
        deadline_unix=args.deadline_unix, seed=20261105, repeat_seeds=[20261106, 20261107],
        initial_candidates=configs, ar_reference_id=ar_id,
        target={"max_accuracy_gap":args.max_accuracy_gap,
                "accuracy_order":"AR > DARD > standard",
                "latency_order":"standard < DARD < AR",
                "latency_metric":"mean generation seconds, all cases",
                "reference_policy":"Fixed prior AR best-quality settings; measured again concurrently",
                "native_selection":"Highest accuracy native in same shared context faster than DARD; retain full native curve and quality references",
                "evidence":"Exploratory validation operating points, not confirmed statistical noninferiority"},
        scope="Current pinned AR and base DiffuCoder; alternate models permitted only after this feasibility check",
        selection="Freeze shortlist after validation, repeat twice; never select from smoke accuracy or exposed holdout",
        training="No weight updates", failure_limit=1,
        monitoring="30-second GPU telemetry and controller heartbeat; no recursive artifact inventory",
        limitations=["Existing validation is previously exposed; no fresh unseen generalization claim",
                     "A failed bounded search does not prove that the model pair can never satisfy the target",
                     "High-confidence validation is not functional correctness verification"])
    freeze_json(plan_path, plan)
    write_json(output / "target_contract.json", plan["target"])
    update(plan, "prepared", status="prepared", candidate_count=len(configs),
           expected_cases=len(configs)*len(read_jsonl(Path(plan["data"])/"validation_tasks.jsonl")),
           child_pids=[])
    return plan


def telemetry(plan, stop):
    destination = Path(plan["output"])
    samples = errors = 0
    while not stop.is_set():
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                    timeout=10, check=True)
            row = {"time_unix":time.time(), "devices_csv":result.stdout.strip(),
                   "scope":"Whole device; one worker per GPU"}
            with (destination/"gpu_samples.jsonl").open("a") as stream:
                stream.write(json.dumps(row)+"\n")
            samples += 1
            write_json(destination/"monitor_status.json", {"status":"running", "updated_unix":time.time(),
                "samples_this_process":samples, "errors_this_process":errors, "last_sample":row})
        except Exception as exc:
            errors += 1
            write_json(destination/"monitor_status.json", {"status":"degraded", "updated_unix":time.time(),
                "samples_this_process":samples, "errors_this_process":errors, "error":repr(exc)})
        stop.wait(30)
    write_json(destination/"monitor_status.json", {"status":"stopped", "updated_unix":time.time(),
                "samples_this_process":samples, "errors_this_process":errors})


def analyze(plan, phases, name, candidate_ids=None):
    from tradeoff_analysis import analyze_evaluations
    from target_tradeoff_search import evaluate_target
    out = Path(plan["output"])/"analysis"/name
    report = analyze_evaluations([Path(plan["output"])/p for p in phases], output=out,
        candidate_ids=candidate_ids, bootstrap_repetitions=2000, bootstrap_seed=20261105,
        quality_tolerance=plan["target"]["max_accuracy_gap"], write_svg=True)
    assessment = evaluate_target(report["summaries"], plan["ar_reference_id"],
        max_accuracy_gap=plan["target"]["max_accuracy_gap"], comparison_mode="matched")
    write_json(out/"target_assessment.json", assessment)
    return report, assessment


def finalists(plan, report, assessment=None):
    """Repeat fixed AR and the strongest near-target DARD, plus fair native pair.

    All pilot candidates remain reported, including higher quality/slower native.
    This is a bounded follow-up, not a proof of global infeasibility.
    """
    valid = [r for r in report["summaries"] if r["valid_for_selection"]]
    ar = next(r for r in valid if r["candidate_id"] == plan["ar_reference_id"])
    dards = [r for r in valid if r["candidate"]["decoder"] == "dard"]
    below_ar = [r for r in dards if r["mean_generation_seconds"] < ar["mean_generation_seconds"]]
    quality = lambda r: (-r["pass_rate"], r["mean_generation_seconds"], r["candidate_id"])
    # Preserve both the latency-eligible leader and absolute DARD quality leader.
    choices = {min(dards,key=quality)["candidate_id"]: min(dards,key=quality)}
    if below_ar:
        best = min(below_ar,key=quality)
        choices[best["candidate_id"]] = best
    # A lower-quality DARD may be the only one with a suitable fast native.
    # Preserve the actually feasible operating point, not only mode maxima.
    if assessment and assessment.get("feasible_triads"):
        feasible = assessment["feasible_triads"][0]
        selected = next(r for r in dards if r["candidate_id"] == feasible["dard_id"])
        choices[selected["candidate_id"]] = selected
    chosen = {ar["candidate_id"]: ar["candidate"]}
    for dard in choices.values():
        chosen[dard["candidate_id"]] = dard["candidate"]
        natives = [r for r in valid if r["candidate"]["decoder"] == "standard"
                   and shared_context(r["candidate"]) == shared_context(dard["candidate"])]
        if not natives:
            raise ValueError("DARD shortlist has no matched native context")
        # Repeat the complete native curve in every selected shared context.
        # A changed timing/ranking on another seed must not weaken the baseline.
        for native in natives:
            chosen[native["candidate_id"]] = native["candidate"]
    return list(chosen.values())


def execute(plan):
    output = Path(plan["output"])
    with (output/"controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_plan(plan)
        if read_json(output/"iteration_state.json",{}).get("status") == "completed":
            return
        if time.time() >= plan["deadline_unix"]-120:
            raise TimeoutError("Search deadline reached")
        stop = threading.Event()
        monitor = threading.Thread(target=telemetry, args=(plan,stop), daemon=True)
        monitor.start()
        try:
            # Smoke is only for execution/coverage diagnostics, never selection.
            evaluate(plan, "smoke", plan["initial_candidates"], split="smoke",
                     tasks_override=Path(plan["data"])/"smoke_tasks.jsonl")
            evaluate(plan, "validation_grid", plan["initial_candidates"])
            report, preliminary = analyze(plan,["validation_grid"],"grid")
            shortlist = finalists(plan,report,preliminary)
            freeze_json(output/"shortlist.json", {"candidates":shortlist,
                "selection_uses":"validation_only", "grid_sha256":file_sha(output/"validation_grid/results.json")})
            phases = ["validation_grid"]
            for seed in plan["repeat_seeds"]:
                phase = "validation_seed_"+str(seed)
                evaluate({**plan,"seed":seed},phase,shortlist)
                phases.append(phase)
            final_report, assessment = analyze(plan,phases,"finalists",[c["id"] for c in shortlist])
            write_json(output/"target_result.json", {"status":"completed", "completed_unix":time.time(),
                "target":plan["target"], "assessment":assessment,
                "scope":"Current pinned model pair; finite exploratory validation search",
                "next_if_unmet":"Inspect refinement failures; consider permitted alternate model pairs. Do not claim general impossibility.",
                "analysis_path":str(output/"analysis/finalists"), "plan_sha256":file_sha(output/"iteration_plan.json")})
            update(plan,"completed",status="completed",child_pids=[],ended_unix=time.time())
        except BaseException as exc:
            write_json(output/"failure.json", {"time_unix":time.time(),"error_type":type(exc).__name__,
                "message":str(exc),"traceback":traceback.format_exc()})
            update(plan,"failed",status="failed",child_pids=[],error_type=type(exc).__name__)
            raise
        finally:
            stop.set()
            monitor.join(timeout=12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path, default=Path("/data/hybrid-dard/results/code-generation-tradeoff-20261107"))
    parser.add_argument("--max-accuracy-gap", type=float, default=.05)
    parser.add_argument("--deadline-unix", type=float, default=1794038400)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--post-stop", action="store_true")
    args = parser.parse_args()
    if not 0 < args.max_accuracy_gap < 1:
        raise ValueError("Accuracy gap must be a fraction in (0,1)")
    if args.prepare:
        plan = prepare(args)
        print(json.dumps({"prepared":True,"candidates":len(plan["initial_candidates"]),"target":plan["target"]}))
        return
    plan = read_json(args.output/"iteration_plan.json")
    if args.post_stop:
        state = read_json(args.output/"iteration_state.json", {})
        if state.get("status") != "completed":
            write_json(args.output/"service_exit.json", {"time_unix":time.time(),
                "service_result":os.environ.get("SERVICE_RESULT", "unknown"),
                "exit_code":os.environ.get("EXIT_CODE"),"exit_status":os.environ.get("EXIT_STATUS"),
                "last_stage":state.get("stage")})
            if state.get("status") not in ("failed", "completed"):
                update(plan,"process_lost",status="failed",child_pids=[],error_type="ServiceExitedBeforeCompletion")
        return
    verify_plan(plan)
    if not args.execute:
        print(json.dumps({"verified":True,"candidates":len(plan["initial_candidates"])}))
        return
    def interrupted(signum, frame):
        raise InterruptedError("Target search interrupted")
    signal.signal(signal.SIGTERM,interrupted)
    with owned_descendant_cleanup():
        execute(plan)


if __name__ == "__main__":
    main()
