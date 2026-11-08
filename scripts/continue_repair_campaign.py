#!/usr/bin/env python3
"""Resume a deadline-interrupted repair campaign without regenerating completed cases."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iterative_tuning import file_sha, read_json, write_json
from scripts.four_gpu import owned_descendant_cleanup
from scripts.iterative_experiments import evaluate, verify_plan
from scripts.run_repair_campaign import (_final_native_controls, _rank_repaired_points,
                                          _validate_model_gate)
from scripts.target_tradeoff_experiments import analyze


def _copy_partial_validation(output: Path):
    source = output / "validation"
    resumed = output / "validation_resume"
    if resumed.exists():
        return resumed
    if not source.is_dir():
        raise FileNotFoundError("Partial validation directory is missing")
    resumed.mkdir(parents=True, exist_ok=True)
    for name in ("cases", "generations"):
        src = source / name
        if src.is_dir():
            shutil.copytree(src, resumed / name)
    return resumed


def run(args):
    output = args.output.resolve()
    plan = read_json(output / "iteration_plan.json")
    verify_plan(plan)
    # The frozen v7 plan remains immutable; only the in-memory execution
    # deadline is extended for this explicitly requested continuation.
    execution = copy.deepcopy(plan)
    execution["deadline_unix"] = args.deadline_unix
    execution["final_reserve_seconds"] = args.final_reserve_seconds
    source_validation = output / "validation" / "evaluation_plan.json"
    if not source_validation.is_file():
        raise FileNotFoundError("validation/evaluation_plan.json is missing")
    validation_plan = read_json(source_validation)
    candidates = validation_plan["candidates"]
    resumed_dir = _copy_partial_validation(output)
    # evaluate() will recover existing case files and only run missing cases.
    evaluate(execution, "validation_resume", candidates, split="validation",
             deadline=args.deadline_unix - args.final_reserve_seconds)
    validation_report, _ = analyze(execution, ["validation_resume"], "validation_resume")
    validation_scores, ranked = _rank_repaired_points(
        validation_report, execution["ar_reference_id"], 1000)
    target = execution["final_target_context"]
    target_ranked = [p for p in ranked
                     if p["candidate"].get("tokens") == target["tokens"]
                     and p["candidate"].get("prefix") == target["prefix"]
                     and p["candidate"].get("dard", {}).get("temperature") == target["temperature"]
                     and p["candidate"].get("dard", {}).get("top_p") == target["top_p"]]
    points = (target_ranked or ranked)[:2]
    if not points:
        raise RuntimeError("No repaired DARD candidate survived resumed validation")
    final = {c["id"]: c for c in candidates
             if c["decoder"] == "ar" and c.get("tokens") == target["tokens"]}
    for point in points:
        dard = point["candidate"]
        final[dard["id"]] = dard
        for candidate in candidates:
            if (candidate["decoder"] == "standard"
                    and candidate["model"] == dard["model"]
                    and candidate["revision"] == dard["revision"]
                    and candidate["tokens"] == dard["tokens"]
                    and candidate["prefix"] == dard["prefix"]
                    and candidate["dard"]["temperature"] == dard["dard"]["temperature"]
                    and candidate["dard"]["top_p"] == dard["dard"]["top_p"]):
                final[candidate["id"]] = candidate
        for candidate in _final_native_controls(dard):
            final[candidate["id"]] = candidate
    final = list(final.values())
    write_json(output / "final_shortlist_resume.json", {
        "candidates": final, "validation_scores": validation_scores,
        "source": "validation_resume", "seeds": plan["repeat_seeds"]})
    phases = []
    for seed in plan["repeat_seeds"]:
        phase = f"final_resume_{seed}"
        evaluate({**execution, "seed": seed}, phase, final, split="validation",
                 deadline=args.deadline_unix - 120)
        phases.append(phase)
    final_report, final_assessment = analyze(execution, phases, "final_resume")
    holdout = evaluate(execution, "holdout_resume", final, split="holdout",
                       deadline=args.deadline_unix - 60)
    holdout_manifest = {
        "schema_version": 1, "split": "holdout", "selection_used": False,
        "evaluation_plan": str(output / "holdout_resume" / "evaluation_plan.json"),
        "evaluation_plan_sha256": file_sha(output / "holdout_resume" / "evaluation_plan.json"),
        "results_path": str(output / "holdout_resume" / "results.json"),
        "results_sha256": file_sha(output / "holdout_resume" / "results.json"),
        "candidate_ids": [c["id"] for c in final], "candidate_count": len(holdout)}
    write_json(output / "holdout_manifest_resume.json", holdout_manifest)
    gate, gate_check = _validate_model_gate(execution, output)
    target_context_matched = any(
        c["decoder"] == "dard" and c.get("tokens") == target["tokens"]
        and c.get("prefix") == target["prefix"]
        and c.get("dard", {}).get("temperature") == target["temperature"]
        and c.get("dard", {}).get("top_p") == target["top_p"] for c in final)
    result = {
        "status": "completed", "completed_unix": time.time(),
        "target": execution["target"], "analysis_path": str(output / "analysis/final_resume"),
        "final_scores": validation_scores, "final_assessment": final_assessment,
        "final_seeds": execution["repeat_seeds"],
        "holdout_manifest_path": str(output / "holdout_manifest_resume.json"),
        "model_gate": gate, "model_gate_check": gate_check,
        "target_context_matched": target_context_matched,
        "target_claim_allowed": gate_check["passed"] and target_context_matched,
        "resumed_from": str(source_validation), "deadline_unix": args.deadline_unix}
    write_json(output / "target_result_resume.json", result)
    write_json(output / "resume_state.json", {"status": "completed", "ended_unix": time.time(),
        "deadline_unix": args.deadline_unix, "reused_partial_validation": True})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--final-reserve-seconds", type=int, default=900)
    args = parser.parse_args()
    try:
        with owned_descendant_cleanup():
            result = run(args)
        print(json.dumps({"status": result["status"], "target_claim_allowed": result["target_claim_allowed"]}))
    except BaseException as exc:
        output = args.output.resolve()
        write_json(output / "resume_failure.json", {"status": "failed", "time_unix": time.time(),
            "error_type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc()})
        write_json(output / "resume_state.json", {"status": "failed", "time_unix": time.time(),
            "error_type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
