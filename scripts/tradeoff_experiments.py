#!/usr/bin/env python3
"""Frozen-weight, decoder-blind four-GPU speed/quality experiments.

Validation selects settings. Previously exposed holdout is descriptive only.
Every generation, including incomplete outputs and budget failures, is retained.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iterative_tuning import (AR, CPGRPO, adapter_identity, candidate, digest, file_sha,
    freeze_json, read_json, read_jsonl, validate_candidate, write_json, task_id, shard_tasks)
from iteration_observability import log_event
from scripts.four_gpu import resolve_gpus, owned_descendant_cleanup
from scripts.iterative_experiments import (AlreadyRunning, evaluate, source_files,
                                          update, verify_plan)


CONTEXTS = ((256, "none", 0., 1.), (256, "code_fence", 0., 1.),
            (256, "none", .3, .95), (256, "none", .4, .95),
            (512, "none", 0., 1.), (512, "code_fence", .3, .95))


def configured(*, adapter=None, decoder="dard", tokens=256, prefix="none",
               temperature=0., top_p=1., steps=32, block=128, tau_c=.5, tau_u=.8,
               completion_policy="legacy_force", diagnostics="summary"):
    value = candidate(CPGRPO, decoder=decoder, tokens=tokens, prefix=prefix,
        temperature=temperature, top_p=top_p, native_steps=steps if decoder == "standard" else 1,
        steps=steps, block=block, tau_c=tau_c, tau_u=tau_u,
        forward_budget=512, adapter=adapter)
    value["dard"].update(completion_policy=completion_policy,
                         completion_verification_steps=2,
                         diagnostics=diagnostics,
                         max_step_records=512 if diagnostics == "steps" else 0)
    value.pop("id")
    value["id"] = decoder + "-" + digest(value)[:16]
    validate_candidate(value)
    return value


def make_grid(adapter):
    """Symmetric shared contexts, plus preregistered DARD-only coordinates."""
    configs = {}
    def add(**kwargs):
        value = configured(**kwargs)
        configs[value["id"]] = value
    for weights in (None, adapter):
        for tokens, prefix, temperature, top_p in CONTEXTS:
            shared = dict(adapter=weights, tokens=tokens, prefix=prefix,
                          temperature=temperature, top_p=top_p)
            for steps in (32, 64, 128, 256):
                add(**shared, decoder="standard", steps=steps)
            for steps in (16, 32, 64):
                for policy in ("legacy_force", "force_then_verify"):
                    add(**shared, steps=steps, completion_policy=policy)
            add(**shared, steps=64, completion_policy="verified_only")
        # Paper language-task threshold grid, including the equality boundary.
        for tau_c in (.4, .5, .6):
            for tau_u in (.6, .7, .8):
                for policy in ("legacy_force", "force_then_verify"):
                    add(adapter=weights, steps=64, tau_c=tau_c, tau_u=tau_u,
                        completion_policy=policy)
        for block in (64, 256):
            for policy in ("legacy_force", "force_then_verify"):
                add(adapter=weights, block=block, steps=32, completion_policy=policy)
        for policy in ("legacy_force", "force_then_verify"):
            add(adapter=weights, steps=128, completion_policy=policy)
    for tokens in (256, 512):
        value = candidate(AR, decoder="ar", tokens=tokens, native_steps=1,
                          temperature=0, top_p=1, forward_budget=512, prefix="code_fence")
        configs[value["id"]] = value
    return list(configs.values())


def shared_context(config):
    return (config["model"], config["revision"], digest(config.get("adapter")),
            config["tokens"], config["prefix"], config["dard"]["temperature"],
            config["dard"]["top_p"])


def choose_finalists(results):
    """At most two shared contexts per weight set, preserving both decoders.

Choose the context containing each decoder's quality maximum. Within each
context retain each decoder's quality maximum and fastest within 3 points.
No holdout data or success-only timings enter this deterministic rule.
"""
    groups = defaultdict(list)
    selected = {}
    for row in results:
        c = row["candidate"]
        if c["decoder"] == "ar":
            selected[c["id"]] = c
        else:
            groups[(c["model"], c["revision"], digest(c.get("adapter")))].append(row)
    def best(rows):
        return min(rows, key=lambda r: (-r["pass_rate"], r["mean_generation_seconds"], r["candidate"]["id"]))
    for rows in groups.values():
        contexts = {shared_context(best([r for r in rows if r["candidate"]["decoder"] == mode])["candidate"])
                    for mode in ("standard", "dard")}
        for context in sorted(contexts):
            for mode in ("standard", "dard"):
                eligible = [r for r in rows if shared_context(r["candidate"]) == context
                            and r["candidate"]["decoder"] == mode]
                if not eligible:
                    raise ValueError("A shortlisted context lacks its paired decoder")
                quality = best(eligible)
                fast = min((r for r in eligible if r["pass_rate"] >= quality["pass_rate"]-.03),
                           key=lambda r: (r["mean_generation_seconds"], -r["pass_rate"], r["candidate"]["id"]))
                for value in (quality, fast):
                    c = value["candidate"]
                    selected[c["id"]] = c
    return sorted(selected.values(), key=lambda c: c["id"])


def smoke_candidates(configs):
    output = [next(c for c in configs if c["decoder"] == "ar" and c["tokens"] == 256)]
    for mode, policy in (("standard", "legacy_force"), ("dard", "legacy_force"),
                         ("dard", "force_then_verify"), ("dard", "verified_only")):
        output.append(next(c for c in configs if c["decoder"] == mode and c["adapter"] is None
            and c["tokens"] == 256 and c["prefix"] == "none" and c["dard"]["temperature"] == 0
            and c["dard"]["completion_policy"] == policy and c["dard"]["tau_c"] == .5
            and c["dard"]["tau_u"] == .8 and
            (c["standard_steps"] == 32 if mode == "standard" else c["dard"]["steps_per_block"] == 64)))
    output.append(next(c for c in configs if c["decoder"] == "dard" and c["adapter"] is not None
        and c["dard"]["tau_c"] == c["dard"]["tau_u"] == .6
        and c["dard"]["completion_policy"] == "force_then_verify"))
    return output


def make_smoke_tasks(validation):
    """Keep four tasks inside one family/language stratum for every GPU.

    shard_tasks stratifies by BOTH family and language. Merely choosing four
    per language can leave shard 3 empty when those four span smaller families.
    """
    small = []
    for language in ("py", "js", "cpp", "rs"):
        families = defaultdict(list)
        for row in validation:
            if row["language"] == language:
                families[row["family"]].append(row)
        eligible = sorted(f for f, rows in families.items() if len(rows) >= 4)
        if not eligible:
            raise ValueError("Each language requires a family with four smoke tasks")
        small.extend(sorted(families[eligible[0]], key=lambda t: digest(t["source_id"]))[:4])
    for index in range(4):
        rows = shard_tasks(small, index)
        if len(rows) != 4 or {r["language"] for r in rows} != {"py", "js", "cpp", "rs"}:
            raise ValueError("Smoke must give each GPU one task in each language")
    return small


def prepare(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "iteration_plan.json"
    if path.exists():
        value = read_json(path)
        verify_plan(value)
        return value
    prior = args.previous.resolve()
    old_plan = read_json(prior / "iteration_plan.json")
    if not old_plan:
        raise ValueError("Previous split and model-integrity receipts are required")
    data = output / "data"
    data.mkdir(exist_ok=True)
    for name in ("train_tasks.jsonl", "validation_tasks.jsonl", "holdout_tasks.jsonl",
                 "split_manifest.json", "data_accounting.json"):
        shutil.copy2(prior / "data" / name, data / name)
    validation = read_jsonl(data / "validation_tasks.jsonl")
    train_sources = {t["source_id"] for t in read_jsonl(data / "train_tasks.jsonl")}
    holdout_sources = {t["source_id"] for t in read_jsonl(data / "holdout_tasks.jsonl")}
    if train_sources & {t["source_id"] for t in validation} or holdout_sources & {t["source_id"] for t in validation}:
        raise ValueError("Training/validation/holdout source overlap")
    small = make_smoke_tasks(validation)
    (data / "smoke_tasks.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False, sort_keys=True)+"\n" for t in small))
    adapter = adapter_identity(prior / "round-01/training/checkpoints/step-00000100")
    configs = make_grid(adapter)
    snapshot = output / "source"
    hashes = {}
    for source in source_files(ROOT):
        relative = source.relative_to(ROOT)
        dest = snapshot / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        hashes[str(relative)] = file_sha(dest)
    plan = {"schema_version": 2, "campaign": "dard_matched_tradeoff_v1", "output": str(output),
        "source": str(snapshot), "source_sha256": hashes, "data": str(data),
        "data_root": old_plan["data_root"], "python": old_plan["python"],
        "gpus": resolve_gpus(args.gpus), "deadline_unix": args.deadline_unix,
        "grading_timeout": 300, "generation_timeout": 120, "seed": 20261105,
        "repeat_seeds": [20261106, 20261107], "failure_limit": 3,
        "data_sha256": {p.name: file_sha(p) for p in data.iterdir() if p.is_file()},
        "previous_plan": {"path": str(prior / "iteration_plan.json"), "sha256": file_sha(prior / "iteration_plan.json")},
        "model_receipts": old_plan["model_receipts"],
        "package_versions": {name: importlib.metadata.version(name) for name in old_plan["package_versions"]},
        "initial_candidates": configs, "smoke_candidates": smoke_candidates(configs),
        "evaluation_options": {"extraction_version": "v2", "counterbalanced_order": True,
                               "candidate_warmup": True, "save_grading_verdict": True},
        "training": "Frozen weights: base cpGRPO and the prior round-01 adapter. No gradient updates.",
        "selection": "All validation cases, then frozen shortlist and two additional seeds. Source-cluster uncertainty, Pareto frontier and latency-budget operating points. No global-optimum claim.",
        "finalist_rule": "Per weights: contexts of standard and DARD quality maxima, both decoders quality maximum and fastest within 3 points; maximum 18 including two AR references.",
        "holdout_policy": "Previously exposed holdout is descriptive replication only. Freeze selection first; do not tune or select from its results.",
        "grading_policy": "Decoder-blind extraction v2 and identical functional tests. Incomplete/verification flags are separate diagnostics and never override functional verdicts.",
        "paper": "https://arxiv.org/html/2608.26574v1",
        "limitations": ["Dream adaptation is not an exact LLaDA reference reproduction.",
                        "Validation and holdout were previously inspected; no fresh unseen generalization claim.",
                        "Shortlist confidence intervals are post-selection, without multiplicity correction.",
                        "Native and DARD forward passes have different cost; wall-clock comparisons are primary."]}
    freeze_json(path, plan)
    write_json(output / "experiment_design.json", {k: plan[k] for k in
        ("campaign", "training", "selection", "finalist_rule", "holdout_policy", "grading_policy", "paper", "limitations")})
    log_event(output, "tradeoff.plan_frozen", plan_sha256=file_sha(path), candidates=len(configs),
              validation_cases=len(validation), source_files=len(hashes))
    return plan


def analyze(plan, phases, name, candidate_ids=None):
    from tradeoff_analysis import analyze_evaluations
    return analyze_evaluations([Path(plan["output"]) / phase for phase in phases],
        output=Path(plan["output"]) / "analysis" / name, candidate_ids=candidate_ids,
        bootstrap_repetitions=2000, bootstrap_seed=20261105, write_svg=True)


def frozen_record(path, value):
    # Do not include wall-clock fields in immutable resumable decisions.
    freeze_json(path, value)
    return value


def selection_rows(report, expected_ids):
    """Use audited case artifacts, never an unchecked cached aggregate."""
    summaries = report["summaries"]
    if isinstance(summaries, dict):
        summaries = list(summaries.values())
    observed = [r["candidate_id"] for r in summaries]
    if len(observed) != len(set(observed)) or set(observed) != set(expected_ids):
        raise ValueError("Audited candidate coverage differs from the frozen selection cohort")
    if any(not r["valid_for_selection"] for r in summaries):
        raise ValueError("Artifact audit found an invalid or incomplete candidate; selection stopped")
    return [{"candidate": r["candidate"], "cases": r["expected_cases"],
             "passed": r["passed"], "pass_rate": r["pass_rate"],
             "mean_generation_seconds": r["mean_generation_seconds"]} for r in summaries]


def review_smoke(plan):
    output = Path(plan["output"])
    expected = {task_id(t): digest(t) for t in read_jsonl(Path(plan["data"]) / "smoke_tasks.jsonl")}
    prompts = {}
    records = 0
    for config in plan["smoke_candidates"]:
        paths = list((output / "smoke").glob("shard-*/cases/"+config["id"]+"/*.json"))
        rows = [read_json(p) for p in paths]
        if len(rows) != len(expected) or {r["task_key"] for r in rows} != set(expected):
            raise ValueError("GPU smoke task coverage mismatch")
        for row in rows:
            if (row["task_sha256"] != expected[row["task_key"]] or row["config_sha256"] != digest(config)
                    or row["infrastructure_error"] or row["forward_calls"] > config["forward_budget"]
                    or row["status"] in {"ForwardBudgetExceeded", "GenerationDeadlineExceeded", "infra_error", "not_tested"}
                    or not row.get("rendered_prompt_sha256")
                    or file_sha(row["generation_receipt"]) != row["generation_receipt_sha256"]):
                raise ValueError("GPU smoke binding, budget, or infrastructure failure")
            if config["decoder"] == "dard":
                if type(row.get("incomplete")) is not bool or type(row.get("verification_certified")) is not bool:
                    raise ValueError("DARD completion diagnostics missing")
            key = (shared_context(config), row["task_key"])
            if prompts.setdefault(key, row["rendered_prompt_sha256"]) != row["rendered_prompt_sha256"]:
                raise ValueError("Matched decoders received different rendered prompts")
            records += 1
    return {"passed": True, "cases_checked": records, "prompt_identity_checked": True,
            "receipt_and_budget_checked": True, "completion_metadata_checked": True}


def execute(plan):
    output = Path(plan["output"])
    with (output / "controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AlreadyRunning("Tradeoff controller already running") from exc
        verify_plan(plan)
        state = read_json(output / "iteration_state.json", {})
        if state.get("status") == "completed":
            return
        if state.get("failure_count", 0) >= plan["failure_limit"]:
            raise AlreadyRunning("Retry limit exhausted")
        if time.time() >= plan["deadline_unix"]-120:
            raise TimeoutError("Campaign deadline reached")
        from grading_audit import write_audit_report
        audit_path = output / "audit/grading_audit.json"
        audit = read_json(audit_path, {})
        audited_hashes = set(audit.get("source_contract", {}).get("source_sha256", {}).values())
        required_hashes = {plan["source_sha256"][name] for name in
                          ("grading_audit.py", "evaluation.py", "function_benchmarks.py", "scripts/iterative_eval_worker.py")}
        if not audit.get("passed") or not required_hashes <= audited_hashes:
            update(plan, "grading_audit")
            if audit_path.exists():
                shutil.copy2(audit_path, audit_path.with_name("grading_audit_prior_"+file_sha(audit_path)[:16]+".json"))
            write_audit_report(audit_path, data_root=Path(plan["data_root"]), runtime_checks=True)
        audit = read_json(audit_path)
        # Runtime grader fixtures must be explicitly successful before any GPU work.
        if not audit.get("passed"):
            raise RuntimeError("Grading audit did not pass")
        smoke = evaluate(plan, "smoke", plan["smoke_candidates"], split="smoke",
                         tasks_override=Path(plan["data"]) / "smoke_tasks.jsonl")
        for result in smoke:
            if result["cases"] != 16:
                raise ValueError("Smoke coverage incomplete")
        review = review_smoke(plan)
        write_json(output / "audit/gpu_smoke_review.json", {**review,
            "results": smoke, "reviewed_unix": time.time(),
            "scope": "Execution/coverage/decoder-blind grading; not paper-equivalence validation."})
        evaluate(plan, "validation_grid", plan["initial_candidates"])
        grid_report = analyze(plan, ["validation_grid"], "grid")
        results = selection_rows(grid_report, [c["id"] for c in plan["initial_candidates"]])
        shortlist = choose_finalists(results)
        if len(shortlist) > 18:
            raise ValueError("Finalist bound violated")
        frozen_record(output / "shortlist.json", {"candidates": shortlist, "uses": "validation_only",
            "source_results_sha256": file_sha(output / "validation_grid/results.json"),
            "repeat_seeds": plan["repeat_seeds"]})
        phases = ["validation_grid"]
        for seed in plan["repeat_seeds"]:
            repeated = {**plan, "seed": seed}
            phase = f"validation_seed_{seed}"
            evaluate(repeated, phase, shortlist)
            phases.append(phase)
        repeat_report = analyze(plan, phases, "finalists", candidate_ids=[c["id"] for c in shortlist])
        # Aggregate over exactly the same three validation seeds for every finalist.
        aggregate = selection_rows(repeat_report, [c["id"] for c in shortlist])
        expected = len(read_jsonl(Path(plan["data"]) / "validation_tasks.jsonl"))*(1+len(plan["repeat_seeds"]))
        if any(row["cases"] != expected for row in aggregate):
            raise ValueError("Finalist seed/case coverage differs")
        # Report best observed settings independently by decoder. No deployment switch.
        winners = []
        for mode in ("ar", "standard", "dard"):
            rows = [r for r in aggregate if r["candidate"]["decoder"] == mode]
            winners.append(min(rows, key=lambda r: (-r["pass_rate"], r["mean_generation_seconds"], r["candidate"]["id"])))
        selection = frozen_record(output / "final_selection.json", {"uses": "three_seed_validation_only",
            "winners_by_decoder": winners, "all_finalists": aggregate,
            "automatic_deployment_change": False, "holdout_already_exposed": True,
            "analysis_path": str(output / "analysis/finalists")})
        holdout = evaluate(plan, "descriptive_holdout", [r["candidate"] for r in winners], split="holdout")
        # Detailed step records run separately and never enter timing/selection summaries.
        diagnostics = []
        for c in shortlist:
            if c["decoder"] == "dard" and c["id"] == winners[-1]["candidate"]["id"]:
                value = copy.deepcopy(c)
                value["dard"].update(diagnostics="steps", max_step_records=512)
                value.pop("id")
                value["id"] = "dard-"+digest(value)[:16]
                diagnostics.append(value)
        if diagnostics:
            evaluate(plan, "diagnostic_steps", diagnostics, split="smoke",
                     tasks_override=Path(plan["data"]) / "smoke_tasks.jsonl")
        write_json(output / "tradeoff_result.json", {"status": "completed", "selection": selection,
            "descriptive_holdout": holdout, "holdout_used_for_selection": False,
            "holdout_is_new_unseen_test": False, "completed_unix": time.time(),
            "plan_sha256": file_sha(output / "iteration_plan.json")})
        update(plan, "completed", status="completed", ended_unix=time.time(), child_pids=[])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path, default=Path("/data/hybrid-dard/results/code-generation-iterative-20261107"))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--deadline-unix", type=float, default=1794038400)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--post-stop", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare:
        plan = prepare(args)
        print(json.dumps({"prepared": True, "candidates": len(plan["initial_candidates"])}))
        return 0
    plan = read_json(args.output / "iteration_plan.json")
    if not plan:
        raise ValueError("Prepare immutable plan first")
    if args.post_stop:
        log_event(args.output, "tradeoff.service_post_stop", service_result=os.environ.get("SERVICE_RESULT", "success"))
        from experiment_notifications import notify, flush_pending
        state = read_json(args.output / "iteration_state.json", {})
        if os.environ.get("SERVICE_RESULT", "success") != "success" and state.get("status") == "running":
            count = state.get("failure_count", 0)+1
            update(plan, "process_lost", status="failed", failure_count=count)
            notify(args.output, {"event_id": "tradeoff-process-lost-"+str(count),
                "stage": state.get("stage"), "error_type": "TradeoffProcessLost", "output": str(args.output)})
        flush_pending(args.output, max_records=20, deadline_seconds=45)
        return 0
    verify_plan(plan)
    if not args.execute:
        print(json.dumps({"verified": True, "candidates": len(plan["initial_candidates"])}))
        return 0
    def interrupted(signum, frame):
        raise InterruptedError("Tradeoff service interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with owned_descendant_cleanup():
            execute(plan)
    except AlreadyRunning:
        return 75
    except BaseException as exc:
        state = read_json(args.output / "iteration_state.json", {})
        count = state.get("failure_count", 0)+1
        update(plan, "failed", status="failed", failure_count=count, error_type=type(exc).__name__)
        write_json(args.output / "tradeoff_failure.json", {"error_type": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc(), "failed_unix": time.time()})
        log_event(args.output, "tradeoff.controller_failure", error_type=type(exc).__name__,
                  error=str(exc), traceback=traceback.format_exc(), failure_count=count)
        from experiment_notifications import notify
        notify(args.output, {"event_id": "tradeoff-failure-"+str(count), "stage": state.get("stage"),
                            "error_type": type(exc).__name__, "output": str(args.output)})
        return 75 if count >= plan["failure_limit"] or isinstance(exc, TimeoutError) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
