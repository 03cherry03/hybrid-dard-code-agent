#!/usr/bin/env python3
"""Execute the staged DARD completion/speed repair campaign.

Preparation is CPU-only and freezes source, data, model receipts, candidates,
and selection rules. Execution uses the audited four-GPU worker and never reads
the holdout split for selection.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adaptive_target_search import score_candidates
from iterative_tuning import (AR, digest, file_sha, freeze_json, read_json, read_jsonl,
                               write_json)
from scripts.four_gpu import owned_descendant_cleanup
from scripts.iterative_experiments import evaluate, source_files, update, verify_plan
from scripts.repair_target_experiments import FINAL_NATIVE_STEPS, repair_grid
from scripts.target_tradeoff_experiments import analyze
from scripts.tradeoff_experiments import configured


def _dev_tasks(data, output):
    rows = read_jsonl(Path(data) / "validation_tasks.jsonl")
    selected = []
    for language in ("py", "js", "cpp", "rs"):
        group = [row for row in rows if row["language"] == language]
        selected.extend(group[:12])
    if len(selected) != 48:
        raise ValueError("The repair development cohort must contain 12 tasks per language")
    path = Path(output) / "data" / "repair_dev_tasks.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected)
    if path.exists() and path.read_text() != payload:
        raise ValueError("Frozen repair development cohort changed")
    path.write_text(payload)
    return path


def _smoke_tasks(data, output):
    rows = read_jsonl(Path(data) / "validation_tasks.jsonl")
    selected = []
    for language in ("py", "js", "cpp", "rs"):
        selected.extend([row for row in rows if row["language"] == language][:2])
    if len(selected) != 8:
        raise ValueError("The repair smoke cohort must contain 2 tasks per language")
    path = Path(output) / "data" / "repair_smoke_tasks.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected)
    if path.exists() and path.read_text() != payload:
        raise ValueError("Frozen repair smoke cohort changed")
    path.write_text(payload)
    return path


def prepare(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "iteration_plan.json"
    if path.exists():
        plan = read_json(path)
        verify_plan(plan)
        return plan
    previous = read_json(args.previous.resolve() / "iteration_plan.json")
    verify_plan(previous)
    data = Path(previous["data"])
    smoke = _smoke_tasks(data, output)
    dev = _dev_tasks(data, output)
    snapshot = output / "source"
    sources = {}
    for source in source_files(ROOT):
        relative = source.relative_to(ROOT)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        sources[str(relative)] = file_sha(destination)
    configs = repair_grid()
    if len(previous["gpus"]) != 4:
        raise ValueError("A four-GPU predecessor plan is required")
    plan = {key: copy.deepcopy(previous[key]) for key in (
        "data_root", "data", "data_sha256", "python", "gpus", "package_versions",
        "model_receipts", "grading_timeout", "generation_timeout")}
    plan.update({"schema_version": 1, "campaign": "dard_completion_speed_repair_v1",
        "output": str(output), "source": str(snapshot), "source_sha256": sources,
        "previous_plan": {"path": str(args.previous.resolve() / "iteration_plan.json"),
                          "sha256": file_sha(args.previous.resolve() / "iteration_plan.json")},
        "deadline_unix": args.deadline_unix, "seed": 20261105,
        "final_reserve_seconds": args.final_reserve_seconds,
        "gpus": previous["gpus"], "candidate_count": len(configs),
        "initial_candidates": configs,
        "ar_reference_id": next(c["id"] for c in configs if c["decoder"] == "ar" and c["tokens"] == 512),
        "smoke_tasks": str(smoke), "smoke_tasks_sha256": file_sha(smoke),
        "dev_tasks": str(dev), "dev_tasks_sha256": file_sha(dev),
        "evaluation_options": {"extraction_version": "v2", "candidate_warmup": True,
            "counterbalanced_order": True, "save_grading_verdict": True,
            "grading_workers": 8, "grading_queue_size": 64},
        "target": {"max_accuracy_gap": .05, "accuracy_order": "AR > DARD > native",
            "latency_order": "native < DARD < AR", "legacy_force": "diagnostic_only",
            "selection": "functional and certified quality jointly with mean/p95 latency",
            "holdout_selection": False},
        "repeat_seeds": [20261111, 20261112, 20261113],
        "phase_policy": {"smoke": f"8 tasks, all {len(configs)} configs; prune before full dev",
            "dev": "48 tasks, exact AR-matched target context only after smoke admission",
            "validation": "273 tasks, at most four DARD points plus matched native 4/8/16/32/64/128 curves",
            "final": "two DARD operating points and matched controls over three fresh seeds",
            "holdout": "one-shot report after shortlist is frozen; never used for selection",
            "dream": "separate canary; excluded from this cpGRPO campaign"},
        "selection_formula": "certified completion-qualified pass first; then AR/native target violation; then forced/incomplete rate; then mean/p95 latency",
        "final_target_context": {"tokens": 512, "prefix": "code_fence", "temperature": 0.0, "top_p": 1.0},
        "model_gate": {"artifact": str(args.model_gate.resolve()), "required_for_target_claim": True,
                       "expected_model": next(c["model"] for c in configs if c["decoder"] == "dard"),
                       "native_development_pass_minimum": 0.60,
                       "status": "ce_only_model_quality_review_required"}})
    freeze_json(path, plan)
    update(plan, "prepared", status="prepared", child_pids=[], expected_cases=len(configs) * 8)
    return plan


def _phase_report(plan, phase, name):
    return analyze(plan, [phase], name)[0]


def _smoke_survivors(plan, report):
    """Drop only candidates with execution/infrastructure failures in smoke.

    The eight-task smoke cohort is an admission check, not a quality selector;
    pass rates from it must never decide which model is preferred.
    """
    invalid = {row["candidate_id"] for row in report["summaries"]
               if row.get("missing_or_invalid_cases", 0)
               or row.get("infrastructure_error_count", 0)}
    survivors = [c for c in plan["initial_candidates"] if c["id"] not in invalid]
    ar_id = plan["ar_reference_id"]
    if not any(c["id"] == ar_id for c in survivors):
        raise RuntimeError("The fixed AR reference failed the smoke admission check")
    write_json(Path(plan["output"]) / "smoke_selection.json", {
        "candidate_ids": [c["id"] for c in survivors],
        "excluded_candidate_ids": sorted(invalid),
        "selection_uses": "smoke admission only; no quality ranking",
    })
    return survivors


def _target_context_candidates(plan, candidates):
    target = plan["final_target_context"]
    selected = []
    for candidate in candidates:
        if candidate["decoder"] == "ar":
            if candidate.get("tokens") == target["tokens"]:
                selected.append(candidate)
            continue
        dard = candidate.get("dard", {})
        if (candidate.get("tokens") == target["tokens"]
                and candidate.get("prefix") == target["prefix"]
                and dard.get("temperature") == target["temperature"]
                and dard.get("top_p") == target["top_p"]):
            selected.append(candidate)
    if not any(c["decoder"] == "dard" for c in selected):
        raise RuntimeError("Smoke admission left no DARD candidate in the AR-matched target context")
    write_json(Path(plan["output"]) / "development_scope.json", {
        "context": target, "candidate_ids": [c["id"] for c in selected],
        "selection_uses": "fixed target context after infrastructure smoke; no smoke quality ranking"})
    return selected


def _point_quality_key(point, rows_by_id):
    row = rows_by_id[point["candidate_id"]]
    certified = float(row.get("completion_qualified_pass_rate", 0.0) or 0.0)
    forced = float(row.get("forced_case_rate", 1.0) or 1.0)
    incomplete = row.get("incomplete_output_rate")
    incomplete = 1.0 if incomplete is None else float(incomplete)
    distances = point.get("distances", {})
    return (
        -certified,
        float(distances.get("ar_accuracy_gap_excess", 1.0) or 1.0),
        float(distances.get("native_accuracy_gain_shortfall", 1.0) or 1.0),
        forced, incomplete,
        -float(row.get("pass_rate", 0.0) or 0.0),
        float(row.get("mean_generation_seconds", float("inf"))),
        float(row.get("p95_generation_seconds", float("inf"))),
        point["candidate_id"],
    )


def _rank_repaired_points(report, ar_reference_id, max_dard):
    scores = score_candidates(report["summaries"], ar_reference_id, .05)
    rows_by_id = {row["candidate_id"]: row for row in report["summaries"]}
    points = []
    for point in scores["ranked_candidates"]:
        row = rows_by_id.get(point["candidate_id"])
        policy = point.get("candidate", {}).get("dard", {}).get("completion_policy")
        if (row is None or not row.get("valid_for_selection")
                or row.get("missing_or_invalid_cases", 0)
                or row.get("infrastructure_error_count", 0)
                or policy == "legacy_force"):
            continue
        points.append(point)
    points.sort(key=lambda point: _point_quality_key(point, rows_by_id))
    return scores, points[:max_dard]


def _final_native_controls(dard):
    values = []
    for steps in FINAL_NATIVE_STEPS:
        value = configured(decoder="standard", tokens=dard["tokens"],
                           prefix=dard["prefix"],
                           temperature=dard["dard"]["temperature"],
                           top_p=dard["dard"]["top_p"], steps=steps, block=128)
        value["model"], value["revision"] = dard["model"], dard["revision"]
        value.pop("id", None)
        value["id"] = value["decoder"] + "-" + digest(value)[:16]
        values.append(value)
    return values


def _select_validation_configs(plan, report, max_dard=4, selection_name="dev_selection"):
    scores, selected = _rank_repaired_points(report, plan["ar_reference_id"], max_dard)
    if not selected:
        raise RuntimeError("No repaired DARD candidate completed the development screen")
    configs = {c["id"]: c for c in plan["initial_candidates"]
               if c["decoder"] == "ar" and c["tokens"] == 512}
    for point in selected:
        dard = point["candidate"]
        configs[dard["id"]] = dard
        for row in report["summaries"]:
            candidate = row["candidate"]
            if candidate["decoder"] == "standard" and (
                    candidate["model"], candidate["revision"], candidate["tokens"],
                    candidate["prefix"], candidate["dard"]["temperature"],
                    candidate["dard"]["top_p"]) == (
                    dard["model"], dard["revision"], dard["tokens"], dard["prefix"],
                    dard["dard"]["temperature"], dard["dard"]["top_p"]):
                configs[candidate["id"]] = candidate
        for candidate in _final_native_controls(dard):
            configs[candidate["id"]] = candidate
    write_json(Path(plan["output"]) / (selection_name + ".json"), {
        "candidate_ids": list(configs), "dard_ids": [p["candidate_id"] for p in selected],
        "selection_uses": "development cohort only; certified/forced/incomplete/latency-aware",
        "scores": scores})
    return list(configs.values())


def _validate_model_gate(plan, output):
    gate = read_json(output / plan["model_gate"]["artifact"], {"status": "not_run"})
    reasons = []
    if gate.get("status") != "passed":
        reasons.append("status_not_passed")
    if gate.get("model") != plan["model_gate"]["expected_model"]:
        reasons.append("model_identity_mismatch")
    if gate.get("target_context") != plan["final_target_context"]:
        reasons.append("target_context_mismatch")
    minimum = plan["model_gate"]["native_development_pass_minimum"]
    if float(gate.get("native_development_pass_rate", -1.0)) < minimum:
        reasons.append("native_quality_gate_failed")
    if gate.get("paired_evaluation") is not True:
        reasons.append("paired_evaluation_receipt_missing")
    return gate, {"passed": not reasons, "reasons": reasons}


def execute(plan):
    output = Path(plan["output"])
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_plan(plan)
        smoke_phase = "smoke"
        evaluate(plan, smoke_phase, plan["initial_candidates"], split="validation",
                 tasks_override=plan["smoke_tasks"], deadline=plan["deadline_unix"] - plan["final_reserve_seconds"])
        smoke_report = _phase_report(plan, smoke_phase, "smoke")
        dev_candidates = _target_context_candidates(
            plan, _smoke_survivors(plan, smoke_report))
        dev_phase = "dev"
        evaluate(plan, dev_phase, dev_candidates, split="validation",
                 tasks_override=plan["dev_tasks"], deadline=plan["deadline_unix"] - plan["final_reserve_seconds"])
        report = _phase_report(plan, dev_phase, "dev")
        selected = _select_validation_configs(plan, report)
        freeze_json(output / "validation_selection.json", {"candidates": selected,
            "source_phase": dev_phase, "selection_data": "dev only"})
        evaluate(plan, "validation", selected, split="validation",
                 deadline=plan["deadline_unix"] - plan["final_reserve_seconds"])
        validation_report = _phase_report(plan, "validation", "validation")
        # Select at most two repaired DARD points for expensive repeats, using
        # the same certified-first policy as development selection.
        validation_scores, ranked_points = _rank_repaired_points(
            validation_report, plan["ar_reference_id"], 1000)
        target_context = plan["final_target_context"]
        target_points = [p for p in ranked_points
                         if p["candidate"].get("tokens") == target_context["tokens"]
                         and p["candidate"].get("prefix") == target_context["prefix"]
                         and p["candidate"].get("dard", {}).get("temperature") == target_context["temperature"]
                         and p["candidate"].get("dard", {}).get("top_p") == target_context["top_p"]]
        points = (target_points or ranked_points)[:2]
        if not points:
            raise RuntimeError("No repaired DARD candidate survived full validation")
        final = {c["id"]: c for c in selected if c["decoder"] == "ar" and c["tokens"] == 512}
        for point in points:
            dard = point["candidate"]
            final[dard["id"]] = dard
            for c in selected:
                if c["decoder"] == "standard" and c["model"] == dard["model"] and c["revision"] == dard["revision"] \
                        and c["tokens"] == dard["tokens"] and c["prefix"] == dard["prefix"] \
                        and c["dard"]["temperature"] == dard["dard"]["temperature"] \
                        and c["dard"]["top_p"] == dard["dard"]["top_p"]:
                    final[c["id"]] = c
            for c in _final_native_controls(dard):
                final[c["id"]] = c
        final = list(final.values())
        freeze_json(output / "final_shortlist.json", {"candidates": final,
            "validation_scores": validation_scores, "seeds": plan["repeat_seeds"]})
        phases = []
        for seed in plan["repeat_seeds"]:
            phase = f"final_{seed}"
            evaluate({**plan, "seed": seed}, phase, final, split="validation",
                     deadline=plan["deadline_unix"] - 120)
            phases.append(phase)
        final_report, final_assessment = analyze(plan, phases, "final")
        # Holdout is evaluated exactly once after the shortlist is frozen. It
        # is reported separately and cannot alter the selected candidates.
        holdout = evaluate(plan, "holdout", final, split="holdout",
                           deadline=plan["deadline_unix"] - 60)
        # The validation analyzer intentionally rejects non-validation plans.
        # Keep a cryptographically bound raw holdout receipt instead of feeding
        # holdout scores back through the selection/reporting path.
        write_json(output / "holdout_manifest.json", {
            "schema_version": 1,
            "split": "holdout",
            "selection_used": False,
            "evaluation_plan": str(output / "holdout" / "evaluation_plan.json"),
            "evaluation_plan_sha256": file_sha(output / "holdout" / "evaluation_plan.json"),
            "results_path": str(output / "holdout" / "results.json"),
            "results_sha256": file_sha(output / "holdout" / "results.json"),
            "candidate_ids": [c["id"] for c in final],
            "candidate_count": len(holdout),
        })
        model_gate, model_gate_check = _validate_model_gate(plan, output)
        target_context = plan["final_target_context"]
        target_context_matched = any(
            c["decoder"] == "dard" and
            c.get("tokens") == target_context["tokens"] and
            c.get("prefix") == target_context["prefix"] and
            c.get("dard", {}).get("temperature") == target_context["temperature"] and
            c.get("dard", {}).get("top_p") == target_context["top_p"]
            for c in final)
        write_json(output / "target_result.json", {
            "status": "completed", "completed_unix": time.time(),
            "target": plan["target"], "analysis_path": str(output / "analysis/final"),
            "final_scores": score_candidates(final_report["summaries"], plan["ar_reference_id"], .05),
            "final_assessment": final_assessment,
            "final_seeds": plan["repeat_seeds"],
            "holdout_manifest_path": str(output / "holdout_manifest.json"),
            "model_gate": model_gate,
            "model_gate_check": model_gate_check,
            "target_claim_allowed": model_gate_check["passed"] and target_context_matched,
            "target_context_matched": target_context_matched})
        update(plan, "completed", status="completed", child_pids=[], ended_unix=time.time())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path, default=Path("/data/hybrid-dard/results/code-generation-tradeoff-20261107"))
    parser.add_argument("--deadline-unix", type=float)
    parser.add_argument("--final-reserve-seconds", type=int, default=900)
    parser.add_argument("--model-gate", type=Path, default=Path("/home/leejwian1955/hybrid-dard-model-upgrade-20261107/model-final/model_gate.json"))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        if args.deadline_unix is None:
            parser.error("--deadline-unix is required with --prepare")
        plan = prepare(args)
        print(json.dumps({"prepared": True, "candidate_count": plan["candidate_count"],
                          "output": plan["output"]}))
        return
    plan = read_json(args.output / "iteration_plan.json")
    if not args.execute:
        verify_plan(plan)
        print(json.dumps({"verified": True, "candidate_count": plan["candidate_count"]}))
        return
    def interrupted(signum, frame):
        raise InterruptedError("repair campaign interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with owned_descendant_cleanup():
            execute(plan)
    except BaseException as exc:
        # Preserve the failure before re-raising so deadline/worker failures
        # cannot silently leave a plan looking merely prepared or running.
        output = Path(plan["output"])
        write_json(output / "failure.json", {
            "schema_version": 1,
            "time_unix": time.time(),
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        })
        try:
            state = read_json(output / "iteration_state.json", {})
            count = int(state.get("failure_count", 0)) + 1
            update(plan, "failed", status="failed", child_pids=[], failure_count=count,
                   error_type=type(exc).__name__, error_message=str(exc))
            from experiment_notifications import notify, flush_pending
            notify(output, {"event_id": "repair-failure-" + str(count),
                            "stage": state.get("stage", "unknown"),
                            "error_type": type(exc).__name__, "output": str(output)})
            flush_pending(output, max_records=20, deadline_seconds=45)
        except BaseException:
            pass
        raise


if __name__ == "__main__":
    main()
