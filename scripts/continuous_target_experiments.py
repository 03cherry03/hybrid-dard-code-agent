#!/usr/bin/env python3
"""Time-bounded adaptive validation optimization with a fixed AR reference."""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adaptive_target_search import (propose_next_candidates, score_candidates,
                                    native_controls, joint_distance)
from iterative_tuning import (CPGRPO, INSTRUCT, DREAM_CODER, file_sha, freeze_json,
                             read_json, read_jsonl, write_json)
from scripts.iterative_experiments import evaluate, source_files, update, verify_plan
from scripts.target_tradeoff_experiments import target_grid, telemetry, analyze
from scripts.tradeoff_experiments import shared_context
from scripts.four_gpu import owned_descendant_cleanup

DEADLINE = 1794051000


def prepare(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "iteration_plan.json"
    if path.exists():
        plan = read_json(path)
        verify_plan(plan)
        return plan
    previous = args.previous.resolve()
    old = read_json(previous / "iteration_plan.json")
    verify_plan(old)
    configs, ar_id = target_grid(read_json(previous / "final_selection.json"))
    hashes = {}
    for source in source_files(ROOT):
        relative = source.relative_to(ROOT)
        dest = output / "source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        hashes[str(relative)] = file_sha(dest)
    plan = {k: copy.deepcopy(old[k]) for k in ("data_root", "data", "data_sha256", "python",
        "gpus", "package_versions", "model_receipts", "grading_timeout", "generation_timeout", "evaluation_options")}
    if len(plan["gpus"]) != 4 or len({g["uuid"] for g in plan["gpus"]}) != 4:
        raise ValueError("Four distinct GPUs are required")
    plan["evaluation_options"].update(grading_workers=8, grading_queue_size=64)
    plan.update(schema_version=1, campaign="continuous_three_way_target_optimization_v1",
        output=str(output), source=str(output / "source"), source_sha256=hashes,
        previous_plan={"path":str(previous / "iteration_plan.json"), "sha256":file_sha(previous / "iteration_plan.json")},
        deadline_unix=DEADLINE, final_reserve_seconds=14400, max_rounds=128,
        seed=20261105, repeat_seeds=[20261108, 20261109, 20261110],
        initial_candidates=configs, ar_reference_id=ar_id,
        target={"max_accuracy_gap":.05, "accuracy_order":"AR > DARD > standard",
            "latency_order":"standard < DARD < AR", "latency_metric":"mean generation seconds over all cases",
            "reference_policy":"Fixed prior best-quality AR; measured in every round and final seed",
            "native_selection":"Highest-accuracy faster native in same model/sampling/context; retain full native curve",
            "optimization":"Continue improving the observed accuracy gap and speed ordering whether the target is met or missed"},
        selection="Validation only; independent round reports; freeze finalists before three new validation seeds",
        model_schedule=[list(CPGRPO), list(INSTRUCT), list(DREAM_CODER)],
        candidate_receipt=str(output / "model_candidates/dream_coder/receipt.json"),
        training="No weight updates", holdout_policy="No holdout evaluation or selection; existing integrity hashes remain checked",
        limitations=["Previously exposed validation; exploratory optimization, not an unseen-generalization claim",
            "Finite deadline-bounded search; no claim of global optimality or model-pair impossibility",
            "High token confidence is not a functional-correctness guarantee",
            "Round screening uses separately measured AR and native controls; final comparisons use identical fresh seeds",
            "Grading concurrency changed to 8/64; do not pool historical 2/4 latency measurements"],
        final_selection_policy="Lowest maximum normalized constraint violation, absolute DARD quality leader, and best observed strict target point if distinct; native 4/8/16/256 and every measured native setting in their contexts")
    freeze_json(path, plan)
    write_json(output / "target_contract.json", plan["target"])
    update(plan, "prepared", status="prepared", child_pids=[], deadline_unix=DEADLINE)
    return plan


def model_plan(plan, model):
    """Admit an optional model only through a complete, pinned integrity receipt."""
    result = copy.deepcopy(plan)
    if tuple(model) != DREAM_CODER:
        if tuple(model) not in (CPGRPO, INSTRUCT):
            raise ValueError("Unapproved model identity")
        return result
    path = Path(plan["candidate_receipt"])
    if not path.is_file():
        return None
    receipt = read_json(path)
    if receipt.get("status") != "verified" or (receipt.get("model"), receipt.get("revision")) != DREAM_CODER:
        raise ValueError("Candidate receipt does not verify the pinned Dream-Coder")
    if not receipt.get("files"):
        raise ValueError("Candidate receipt lacks checked files")
    result["model_receipts"].append({"path":str(path), "sha256":file_sha(path)})
    verify_plan(result)
    return result


def batch_budget(configs, history, task_count):
    """Conservative wall-time estimate from completed rounds, with a cold floor."""
    rates = [h["elapsed_seconds"] / h["case_count"] for h in history if h["case_count"]]
    per_case = max(rates[-3:], default=1.0)
    return max(1200., per_case * len(configs) * task_count * 1.35 + 600.)


def should_search(plan, now, estimated_seconds):
    return now + estimated_seconds < plan["deadline_unix"] - plan["final_reserve_seconds"]


def ranked_history(history):
    """Compare screened operating points, preserving each point's own controls."""
    points = []
    for item in history:
        points += [{**p, "round":item["round"]} for p in item["scores"]["ranked_candidates"]]
    return sorted(points, key=lambda p:(tuple(p["rank_vector"]), -p["pass_rate"],
                                       p["mean_generation_seconds"], p["candidate_id"]))


def final_candidates(plan, history):
    ranked = ranked_history(history)
    if not ranked:
        raise ValueError("No complete valid DARD/AR comparison to repeat")
    comparable = [(joint_distance(p), p) for p in ranked]
    comparable = [(distance, p) for distance, p in comparable if distance is not None]
    closest = min(comparable, key=lambda pair:(pair[0], not pair[1].get("target_met", False), tuple(pair[1]["rank_vector"]),
                   -pair[1]["pass_rate"], pair[1]["mean_generation_seconds"]))[1] if comparable else ranked[0]
    selected = {closest["candidate_id"]:closest}
    feasible = next((p for p in ranked if p.get("target_met") is True), None)
    if feasible is not None:
        selected[feasible["candidate_id"]] = feasible
    quality = min(ranked, key=lambda p:(-p["pass_rate"], p["mean_generation_seconds"]))
    selected[quality["candidate_id"]] = quality
    ar = next(c for c in plan["initial_candidates"] if c["id"] == plan["ar_reference_id"])
    configs = {ar["id"]:ar}
    for point in selected.values():
        configs[point["candidate_id"]] = point["candidate"]
        context = shared_context(point["candidate"])
        # Include all measured native controls in this context, across rounds.
        natives = [r["candidate"] for h in history for r in h["report"]["summaries"]
                   if r["valid_for_selection"] and r["candidate"]["decoder"] == "standard"
                   and shared_context(r["candidate"]) == context]
        if not natives:
            raise ValueError("Final DARD has no comparable native curve")
        # Old configurations can differ in irrelevant DARD metadata. Keep one
        # native execution per actual context/step count, preferring observed IDs.
        native_by_setting = {}
        for c in natives + native_controls(point["candidate"]):
            native_by_setting.setdefault((shared_context(c), c["standard_steps"]), c)
        configs.update({c["id"]:c for c in native_by_setting.values()})
    return list(configs.values())


def final_reserve(plan, history, task_count):
    if not history:
        return plan["final_reserve_seconds"]
    configs = final_candidates(plan, history)
    measurements = {}
    for h in history:
        for row in h["report"]["summaries"]:
            if row["valid_for_selection"]:
                c = row["candidate"]
                key = (shared_context(c), c["decoder"], c["standard_steps"] if c["decoder"] == "standard" else c["id"])
                measurements[key] = max(measurements.get(key, 0), row["mean_generation_seconds"])
    seconds = 0.
    for c in configs:
        key = (shared_context(c), c["decoder"], c["standard_steps"] if c["decoder"] == "standard" else c["id"])
        # A newly required 256-step native control may not yet be measured.
        fallback = max((v for k,v in measurements.items() if k[0] == shared_context(c)), default=2.)
        if c["decoder"] == "standard" and key not in measurements:
            native = [(k[2],v) for k,v in measurements.items() if k[0] == shared_context(c) and k[1] == "standard"]
            fallback = max((v * c["standard_steps"] / steps for steps,v in native), default=32.)
        seconds += measurements.get(key, fallback)
    generation_estimate = 1.35 * seconds * task_count / len(plan["gpus"]) + 600
    return max(plan["final_reserve_seconds"], len(plan["repeat_seeds"]) * generation_estimate)


def save_progress(plan, history, failures, stage):
    output = Path(plan["output"])
    points = ranked_history(history)
    comparable = [(joint_distance(p), p) for p in points]
    comparable = [(distance, p) for distance, p in comparable if distance is not None]
    closest = min(comparable, key=lambda pair:(pair[0], not pair[1].get("target_met", False),
                  tuple(pair[1]["rank_vector"]), -pair[1]["pass_rate"]))[1] if comparable else (points[0] if points else None)
    data = {"stage":stage, "updated_unix":time.time(), "deadline_unix":plan["deadline_unix"],
        "completed_rounds":len(history), "search_should_continue":stage == "adaptive_search",
        "ranking_policy":"Quality gap first, then comparator/gain and latency violations; full Pareto retained",
        "scope":"Provisional screening across separately measured rounds; no pooled cross-round statistics",
        "closest_observed":closest,
        "quality_leader":max(points, key=lambda p:p["pass_rate"]) if points else None,
        "failures":failures, "rounds":[{k:v for k,v in h.items() if k not in ("report", "scores")} for h in history]}
    write_json(output / "search_progress.json", data)
    rows = [{"round":h["round"], "model":h["model"], "cases":h["case_count"],
             "seconds":h["elapsed_seconds"], "best_dard_accuracy":max((p["pass_rate"] for p in h["scores"]["ranked_candidates"]), default=None),
             "closest_gap_pp":h["scores"]["ranked_candidates"][0]["accuracy_gap_pp"] if h["scores"]["ranked_candidates"] else None}
            for h in history]
    if rows:
        with (output / "search_progress.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        # Use the existing report's standalone SVG approach without adding a
        # plotting dependency to the frozen inference environment.
        finite_rows = [r for r in rows if r["closest_gap_pp"] is not None]
        lower = min([0.] + [r["closest_gap_pp"] for r in finite_rows])
        upper = max([10.] + [r["closest_gap_pp"] for r in finite_rows]) + 2
        xmax = max(1, max(r["round"] for r in rows))
        def xy(row):
            return 70 + 660 * row["round"] / xmax, 300 - 240 * (row["closest_gap_pp"]-lower) / (upper-lower)
        coords = " ".join(f"{x:.2f},{y:.2f}" for x,y in map(xy, finite_rows))
        target_y = 300 - 240 * (5-lower) / (upper-lower)
        svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="370" viewBox="0 0 800 370">',
            '<rect width="800" height="370" fill="white"/>',
            '<g font-family="sans-serif" font-size="12" fill="#222">',
            '<text x="70" y="25">AR - quality-leading DARD accuracy (percentage points)</text>',
            '<path d="M70 50 V300 H750" fill="none" stroke="#777"/>',
            f'<path d="M70 {target_y:.2f} H750" stroke="#dc2626" stroke-dasharray="5 4"/>',
            f'<text x="570" y="{target_y-8:.2f}">Requested limit: 5 pp</text>',
            f'<polyline points="{coords}" fill="none" stroke="#2563eb" stroke-width="2"/>']
        for row in finite_rows:
            x,y = xy(row)
            svg += [f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#2563eb"><title>Round {row["round"]}: {row["closest_gap_pp"]:.2f} pp</title></circle>',
                    f'<text x="{x:.2f}" y="320" text-anchor="middle">{row["round"]}</text>']
        for value in (lower, (lower+upper)/2, upper):
            y = 300 - 240 * (value-lower) / (upper-lower)
            svg.append(f'<text x="60" y="{y+4:.2f}" text-anchor="end">{value:.1f}</text>')
        svg += ['<text x="300" y="350">Completed adaptive round</text>', '</g></svg>']
        (output / "search_progress.svg").write_text("\n".join(svg)+"\n")


def check_smoke(rows, configs):
    # evaluate() already rejects missing/duplicate tasks and infrastructure errors.
    # Smoke accuracy is deliberately not an admission condition.
    if (not rows or len(rows) != len(configs)
            or {r["candidate"]["id"] for r in rows} != {c["id"] for c in configs}
            or any(r.get("cases") != 16 or set(r.get("by_language", {})) != {"py", "js", "cpp", "rs"}
                   or any(g["cases"] != 4 for g in r["by_language"].values()) for r in rows)):
        raise RuntimeError("Model smoke lacks complete candidate/language coverage")


def check_final_evidence(report, assessment, configs, task_count, completed_seeds):
    """Publish only complete, receipt-validated evidence for every frozen choice."""
    if not isinstance(assessment, dict) or assessment.get("status") not in ("feasible", "no_feasible"):
        raise ValueError("Final evidence has an invalid target reference or assessment")
    if (not completed_seeds or any(type(seed) is not int or seed < 0 for seed in completed_seeds)
            or len(set(completed_seeds)) != len(completed_seeds)):
        raise ValueError("Final evidence requires distinct completed evaluation seeds")
    expected_ids = [c["id"] for c in configs]
    rows = report.get("summaries") if isinstance(report, dict) else None
    if (not expected_ids or len(set(expected_ids)) != len(expected_ids)
            or not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows)
            or len(rows) != len(expected_ids)
            or {r.get("candidate_id") for r in rows} != set(expected_ids)):
        raise ValueError("Final evidence does not cover every shortlisted candidate exactly once")
    expected_cases = task_count * len(completed_seeds)
    for row in rows:
        if row.get("valid_for_selection") is not True:
            raise ValueError("Final evidence contains an invalid candidate or generation receipt")
        if any(type(row.get(key)) is not int or row[key] != expected_cases
               for key in ("expected_cases", "observed_cases")):
            raise ValueError("Final evidence has incomplete task/seed coverage")
        seeds = row.get("seeds")
        if (not isinstance(seeds, list) or any(type(seed) is not int for seed in seeds)
                or sorted(seeds) != sorted(completed_seeds)):
            raise ValueError("Final evidence has missing or unexpected evaluation seeds")


def execute(plan):
    output = Path(plan["output"])
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_plan(plan)
        if read_json(output / "iteration_state.json", {}).get("status") == "completed":
            return
        stop = threading.Event()
        monitor = threading.Thread(target=telemetry, args=(plan, stop), daemon=True)
        monitor.start()
        checkpoint = read_json(output / "controller_checkpoint.json", {})
        history = checkpoint.get("history", [])
        failures = checkpoint.get("failures", [])
        existing = checkpoint.get("existing", [])
        admitted = {tuple(m) for m in checkpoint.get("admitted", [])}
        quarantined = {tuple(m) for m in checkpoint.get("quarantined", [])}
        def save_checkpoint(next_round):
            write_json(output / "controller_checkpoint.json", {"next_round":next_round,
                "history":history, "failures":failures, "existing":existing,
                "admitted":[list(m) for m in sorted(admitted)],
                "quarantined":[list(m) for m in sorted(quarantined)]})
        final_phases = []
        try:
            task_count = len(read_jsonl(Path(plan["data"]) / "validation_tasks.jsonl"))
            ar = next(c for c in plan["initial_candidates"] if c["id"] == plan["ar_reference_id"])
            for index in range(checkpoint.get("next_round", 0), plan["max_rounds"]):
                if (output / "shortlist.json").exists():
                    break
                label = f"adaptive_round_{index:03d}"
                requested = tuple(plan["model_schedule"][index % len(plan["model_schedule"])])
                # Current models remain usable when the optional download is pending.
                if requested in quarantined:
                    requested = CPGRPO
                try:
                    active = model_plan(plan, requested)
                except Exception as exc:
                    if requested != DREAM_CODER:
                        raise
                    failures.append({"round":index, "model":list(requested), "error":repr(exc), "gate":"integrity_receipt"})
                    quarantined.add(requested)
                    requested = CPGRPO
                    active = model_plan(plan, requested)
                if active is None:
                    requested = CPGRPO
                    active = model_plan(plan, requested)
                if (output / label / "proposal.json").exists() and (output / label / "campaign_plan.json").exists():
                    proposal = read_json(output / label / "proposal.json")
                    active = read_json(output / label / "campaign_plan.json")
                    verify_plan(active)
                    configs = proposal["candidates"]
                    if not any(c["id"] == ar["id"] for c in configs):
                        configs = [ar] + configs
                    model_config = next(c for c in configs if c["decoder"] == "dard")
                    requested = (model_config["model"], model_config["revision"])
                elif index == 0:
                    configs = plan["initial_candidates"]
                    proposal = {"reason":"initial fixed baseline and DARD diagnostics", "candidates":configs}
                else:
                    matching = [h for h in history if tuple(h["model"]) == requested]
                    anchor = (matching or history)[-1]["report"]["summaries"] if history else []
                    proposal = propose_next_candidates(anchor, existing_candidates=existing,
                        ar_id=plan["ar_reference_id"], max_accuracy_gap=.05,
                        round_index=index-1, max_candidates=23, model_override=requested)
                    configs = [ar] + proposal["candidates"]
                    if len(configs) == 1:
                        continue
                estimate = batch_budget(configs, history, task_count)
                reserve = final_reserve(plan, history, task_count)
                timing_plan = {**plan, "final_reserve_seconds":reserve}
                if not (output / label / "results.json").exists() and not should_search(timing_plan, time.time(), estimate):
                    break
                freeze_json(output / label / "proposal.json", proposal)
                freeze_json(output / label / "campaign_plan.json", active)
                existing.extend(configs)
                # A resumed phase keeps its already frozen evaluation deadline.
                prior_evaluation = read_json(output / label / "evaluation_plan.json",
                    read_json(output / (label + "_smoke") / "evaluation_plan.json", {}))
                search_deadline = prior_evaluation.get("deadline_unix", plan["deadline_unix"] - reserve)
                started = time.time()
                try:
                    if requested not in admitted:
                        smoke = evaluate(active, label + "_smoke", configs, split="smoke", deadline=search_deadline,
                            tasks_override=Path(plan["data"]) / "smoke_tasks.jsonl")
                        check_smoke(smoke, configs)
                        admitted.add(requested)
                        write_json(output / label / "compatibility.json", {"status":"gpu_smoke_passed", "model":list(requested),
                            "selection_uses_smoke_accuracy":False, "smoke_results_sha256":file_sha(output / (label + "_smoke") / "results.json")})
                    evaluate(active, label, configs, deadline=search_deadline)
                    report, assessment = analyze(active, [label], label)
                    scores = score_candidates(report["summaries"], plan["ar_reference_id"], .05)
                    write_json(output / "analysis" / label / "optimization_scores.json", scores)
                    if not scores["ranked_candidates"]:
                        raise RuntimeError("No valid AR/DARD comparison in completed round")
                    history.append({"round":index, "phase":label, "model":list(requested),
                        "case_count":len(configs)*task_count, "elapsed_seconds":time.time()-started,
                        "report":report, "scores":scores, "assessment_status":assessment.get("status")})
                    # Target attainment is deliberately not a stopping condition.
                    save_progress(plan, history, failures, "adaptive_search")
                    save_checkpoint(index+1)
                except TimeoutError as exc:
                    failures.append({"round":index, "model":list(requested), "error":repr(exc), "partial_round_excluded":True})
                    save_checkpoint(index+1)
                    break
                except Exception as exc:
                    failures.append({"round":index, "model":list(requested), "error":repr(exc), "traceback":traceback.format_exc(), "partial_round_excluded":True})
                    save_progress(plan, history, failures, "adaptive_search")
                    if requested == DREAM_CODER:
                        quarantined.add(requested)
                        save_checkpoint(index+1)
                        continue
                    raise
            if (output / "shortlist.json").exists():
                shortlist = read_json(output / "shortlist.json")["candidates"]
            else:
                shortlist = final_candidates(plan, history)
                freeze_json(output / "shortlist.json", {"candidates":shortlist, "selection_uses":"completed_validation_rounds_only",
                    "repeat_seeds":plan["repeat_seeds"], "rounds":[h["phase"] for h in history]})
            active = copy.deepcopy(plan)
            if any(c["model"] == DREAM_CODER[0] for c in shortlist):
                active = model_plan(plan, DREAM_CODER)
                if active is None or DREAM_CODER not in admitted:
                    raise ValueError("Final optional model lacks integrity and GPU admission")
            save_progress(plan, history, failures, "final_repeats")
            for seed in plan["repeat_seeds"]:
                label = "final_seed_" + str(seed)
                evaluate({**active, "seed":seed}, label, shortlist, deadline=plan["deadline_unix"]-180)
                final_phases.append(label)
                # Publish complete-seed evidence even if later repetition hits deadline.
                report, assessment = analyze(active, final_phases, "finalists")
                check_final_evidence(report, assessment, shortlist, task_count,
                                     [int(p.rsplit("_", 1)[1]) for p in final_phases])
                write_json(output / "final_repeat_progress.json", {"completed_seeds":[int(p.rsplit("_",1)[1]) for p in final_phases],
                    "planned_seeds":plan["repeat_seeds"], "assessment":assessment})
            if time.time() >= plan["deadline_unix"]:
                raise TimeoutError("Optimization deadline reached before final publication")
            write_json(output / "target_result.json", {"status":"completed", "completed_unix":time.time(),
                "target":plan["target"], "assessment":assessment, "optimization":score_candidates(report["summaries"],plan["ar_reference_id"],.05),
                "completed_search_rounds":len(history), "completed_final_seeds":plan["repeat_seeds"],
                "analysis_path":str(output / "analysis/finalists"), "failures":failures,
                "plan_sha256":file_sha(output / "iteration_plan.json")})
            save_progress(plan, history, failures, "completed")
            update(plan, "completed", status="completed", child_pids=[], ended_unix=time.time())
        except BaseException as exc:
            write_json(output / "failure.json", {"time_unix":time.time(), "error_type":type(exc).__name__,
                "message":str(exc), "traceback":traceback.format_exc(), "completed_final_phases":final_phases})
            save_progress(plan, history, failures, "incomplete")
            update(plan, "failed", status="failed", child_pids=[], error_type=type(exc).__name__)
            raise
        finally:
            stop.set(); monitor.join(timeout=12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path, default=Path("/data/hybrid-dard/results/code-generation-tradeoff-20261107"))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--post-stop", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        plan = prepare(args)
        print(json.dumps({"prepared":True, "deadline_unix":plan["deadline_unix"], "gpus":plan["gpus"]}))
        return
    plan = read_json(args.output / "iteration_plan.json")
    if args.post_stop:
        state = read_json(args.output / "iteration_state.json", {})
        write_json(args.output / "service_exit.json", {"time_unix":time.time(), "service_result":os.environ.get("SERVICE_RESULT", "unknown"),
            "exit_code":os.environ.get("EXIT_CODE"), "exit_status":os.environ.get("EXIT_STATUS"), "last_stage":state.get("stage")})
        if state.get("status") not in ("completed", "failed"):
            update(plan, "process_lost", status="failed", child_pids=[], error_type="ServiceExitedBeforeCompletion")
        return
    verify_plan(plan)
    if not args.execute:
        print(json.dumps({"verified":True, "deadline_unix":plan["deadline_unix"]}))
        return
    def interrupted(signum, frame):
        raise InterruptedError("Optimization service interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    with owned_descendant_cleanup():
        execute(plan)


if __name__ == "__main__":
    main()
