#!/usr/bin/env python3
"""Bounded development selection followed by one disjoint multilingual evaluation.

Existing baseline outputs are never read to choose tuning identities or settings.
Only the new development outcomes select one pinned diffusion checkpoint. The
three-point screening margin is descriptive, not statistical noninferiority.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import all_experiments as pipeline, four_gpu
from scripts.focused_experiments import SUITES, freeze_json
from campaign_reporting import write_goal_report
from scripts.verify_prepared_models import signature, verify_receipt

INSTRUCT = "apple/DiffuCoder-7B-Instruct"
INSTRUCT_REVISION = "4fdd4580064ca5d11808069ce78f88d068753c96"
CPGRPO = "apple/DiffuCoder-7B-cpGRPO"
CPGRPO_REVISION = "98bc42bfd871eeec92fd0a8439ab34f75b405204"
LANGUAGES = ("py", "rs", "js", "cpp")
FAMILIES = ("humaneval", "mbpp")
PARTITION_ALGORITHM = "optimization_identity_partition_v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def partition(catalog, sources_per_family=8):
    """Read source identities only, retaining every language together."""
    if type(sources_per_family) is not int or sources_per_family < 1:
        raise ValueError("Development sources per family must be positive")
    selected = []
    for family in FAMILIES:
        rows = [x for x in catalog["evaluation_sources"] if x["family"] == family]
        if len(rows) <= sources_per_family:
            raise ValueError("Development must leave final sources in both families")
        rows.sort(key=lambda x: (digest(["optimization-dev-v1", 20261105, family, x["source_id"]]), x["source_id"]))
        selected.extend(rows[:sources_per_family])
    identities = {x["source_id"] for x in selected}
    remaining = [x for x in catalog["evaluation_sources"] if x["source_id"] not in identities]
    result = []
    for split, rows in (("dev", selected), ("test", remaining)):
        keys = [key for row in rows for key in row["task_keys"]]
        if any(Counter(key[1] for key in row["task_keys"]) != Counter(LANGUAGES) for row in rows):
            raise ValueError("Every selected source must retain all four languages exactly once")
        result.append({"schema_version": 1, "algorithm": PARTITION_ALGORITHM, "split": split,
            "source_count": len(rows), "task_count": len(keys), "languages": list(LANGUAGES),
            "source_ids_by_family": {f: [x["source_id"] for x in rows if x["family"] == f] for f in FAMILIES},
            "task_keys": keys, "selection_uses_only_identities": True,
            "development_final_disjoint_by_source_id": True,
            "baseline_scope_note": "Existing 360-source baseline is exploratory; final test excludes the new tuning cohort. No baseline correctness or timing records read for partition."})
    return result


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    catalog_path = args.baseline_output / "selection_catalog.json"
    catalog = json.loads(catalog_path.read_text())
    dev, final = partition(catalog, args.development_sources_per_family)
    paths = [args.output / "development_selection.json", args.output / "evaluation_selection.json"]
    for path, manifest in zip(paths, (dev, final), strict=True):
        freeze_json(path, manifest)
    receipt_path = args.output / "partition_receipt.json"
    receipt = {"schema_version": 1, "created_unix": time.time(),
        "baseline_catalog_sha256": file_digest(catalog_path), "baseline_output": str(args.baseline_output),
        "selection_algorithm": PARTITION_ALGORITHM, "seed": 20261105,
        "development_sources_per_family": args.development_sources_per_family,
        "development_source_ids": sorted(x for ids in dev["source_ids_by_family"].values() for x in ids),
        "development_manifest_sha256": file_digest(paths[0]), "evaluation_manifest_sha256": file_digest(paths[1]),
        "outcome_files_read": False, "note": "Prior calibration4sources excluded from both partitions."}
    previous = pipeline.read_json(receipt_path)
    if previous:
        receipt["created_unix"] = previous["created_unix"]
    freeze_json(receipt_path, receipt)
    if args.accuracy_margin != .03:
        raise ValueError("This bounded protocol fixes its descriptive development margin to .03")
    if args.cpgrpo_revision not in (None, CPGRPO_REVISION):
        raise ValueError("Only the reviewed pinned cpGRPO checkpoint is allowed")
    if args.cpgrpo_revision and not args.candidate_model_receipt:
        raise ValueError("cpGRPO requires its separate verified model receipt")
    sources = sorted(ROOT.glob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    sources += sorted((ROOT / "scripts").glob("*.sh")) + sorted(ROOT.glob("requirements*.txt"))
    sources += list(ROOT.glob("native_repository_recipes.json"))
    plan = {"schema_version": 1, "baseline_output": str(args.baseline_output),
        "output": str(args.output), "data_root": str(args.data_root), "gpus": args.gpus,
        "deadline_unix": args.deadline_unix, "development_budget_seconds": args.development_budget_seconds,
        "accuracy_margin": args.accuracy_margin, "selection_criterion": "Among complete DARD development candidates within 0.03 pass fraction of AR and strictly faster mean generation, minimize mean time then maximize accuracy. If none qualify, maximize accuracy then minimize time and explicitly report goal unmet.",
        "screening_is_statistical_noninferiority": False,
        "cpgrpo_revision": args.cpgrpo_revision,
        "candidate_model_receipt": str(args.candidate_model_receipt) if args.candidate_model_receipt else None,
        "optional_model_wait_seconds": args.optional_model_wait_seconds,
        "development_manifest_sha256": receipt["development_manifest_sha256"],
        "evaluation_manifest_sha256": receipt["evaluation_manifest_sha256"],
        "source_sha256": {str(p.relative_to(ROOT)): file_digest(p) for p in sources},
        "settings": {"tokens": 512, "block_size": 512, "dard_steps": 64, "standard_steps": 64,
            "native_quality_diagnostic_steps": 256, "temperature": 0, "top_p": 1,
            "tau_c": .5, "tau_u": .8, "transfer_limit": 0, "repairs": 0, "repeats": 1},
        "final_backends": ["ar", "dard", "standard"], "final_languages": list(LANGUAGES),
        "development_is_disjoint_from_final": True, "prior_baseline_is_exploratory": True}
    model_receipt = args.baseline_output / "audit/model_integrity.json"
    plan["prepared_model_receipt"] = str(model_receipt)
    plan["prepared_model_receipt_sha256"] = file_digest(model_receipt)
    freeze_json(args.output / "optimization_plan.json", plan)
    return plan


def event(args, state, stage, **details):
    state.update(stage=stage, updated_unix=time.time(), **details)
    pipeline.write_json(args.output / "optimization_state.json", state)
    with (args.output / "optimization_events.jsonl").open("a") as stream:
        stream.write(json.dumps({"timestamp_unix": time.time(), "invocation_id": state["invocation_id"],
                                 "stage": stage, **details}, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def bounded(deadline):
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError("Experiment deadline reached")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    def expired(signum, frame):
        raise TimeoutError("Experiment deadline reached")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, min(remaining, previous_timer[0]) if previous_timer[0] else remaining)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0]:
            signal.setitimer(signal.ITIMER_REAL, max(.001, previous_timer[0] - (time.monotonic()-started)), previous_timer[1])


def wait_baseline(args, state):
    event(args, state, "waiting_for_baseline")
    while True:
        if time.time() >= args.deadline_unix - 1800:
            raise TimeoutError("Baseline did not finish before reserved analysis period")
        saved = pipeline.read_json(args.baseline_output / "guard_state.json", {})
        if saved.get("status") == "failed":
            raise RuntimeError("Existing baseline failed; refusing to overlap uncertain GPU ownership")
        pipeline_state = pipeline.read_json(args.baseline_output / "pipeline_state.json", {})
        if (saved.get("status") == "completed" and pipeline_state.get("status") == "completed"
                and gpus_available(args)):
            return
        event(args, state, "waiting_for_baseline", baseline_stage=saved.get("stage"), baseline_status=saved.get("status"))
        time.sleep(15)


def gpus_available(args):
    """Check both coordinator locks and actual GPU ownership, without launching work."""
    import fcntl
    from contextlib import ExitStack
    selected = four_gpu.resolve_gpus(args.gpus)
    locks = args.data_root / "work/gpu-locks"
    locks.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        try:
            for gpu in sorted(selected, key=lambda x: x["uuid"]):
                stream = stack.enter_context((locks / (gpu["uuid"] + ".lock")).open("a"))
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        uuids = {gpu["uuid"] for gpu in selected}
        processes = four_gpu.run_checked(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"])
        return not any(line.split(",")[0].strip() in uuids for line in processes.splitlines())


def validate_gate(args):
    output = args.output / "validation"
    script = ROOT / "scripts/validate_dard_optimization.py"
    if not script.is_file():
        raise RuntimeError("Required numerical/GPU validation script is missing")
    command = [str(args.data_root / "envs/inference/bin/python"), str(script),
        "--data-root", str(args.data_root), "--gpus", args.gpus, "--output", str(output)]
    old = pipeline.read_json(output / "validation_report.json", {})
    # Validation cannot be skipped based solely on an unbound passed flag.
    marker = args.output / "validation_gate.json"
    existing = pipeline.read_json(marker)
    expected = {"command": command, "script_sha256": file_digest(script)}
    if existing and existing.get("binding") == expected and old.get("status") == "passed" and existing.get("report_sha256") == file_digest(output / "validation_report.json"):
        return existing
    output.mkdir(parents=True, exist_ok=True)
    with (output / "validation.log").open("a") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=pipeline.runtime_env(args.data_root),
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=min(1800, max(1, args.deadline_unix-time.time()-1800)))
        except BaseException:
            four_gpu._terminate_workers([(0, child)])
            raise
    report = pipeline.read_json(output / "validation_report.json", {})
    if code or report.get("status") != "passed":
        raise RuntimeError("DARD numerical/GPU validation failed")
    record = {"binding": expected, "report_sha256": file_digest(output / "validation_report.json"),
              "status": "passed", "completed_unix": time.time()}
    pipeline.write_json(marker, record)
    return record


def candidate_available(args, state, dev_deadline):
    if not args.cpgrpo_revision:
        return False
    decision_path = args.output / "candidate_model_decision.json"
    previous = pipeline.read_json(decision_path)
    if previous:
        if previous.get("status") == "skipped":
            return False
        verify_candidate_receipt(args)
        return True
    wait_until = min(dev_deadline - 1800, time.time()+args.optional_model_wait_seconds)
    while not args.candidate_model_receipt.is_file() and time.time() < wait_until:
        event(args, state, "waiting_for_optional_model")
        time.sleep(min(15, max(.01, wait_until-time.time())))
    if not args.candidate_model_receipt.is_file():
        pipeline.write_json(decision_path, {"status": "skipped", "reason": "Optional checkpoint not ready within reserved development time", "decided_unix": time.time()})
        return False
    info = verify_candidate_receipt(args)
    pipeline.write_json(decision_path, {"status": "verified", "decided_unix": time.time(), **info})
    return True


def verify_candidate_receipt(args):
    receipt = json.loads(args.candidate_model_receipt.read_text())
    if (receipt.get("status") != "verified" or receipt.get("model") != CPGRPO
            or receipt.get("revision") != args.cpgrpo_revision or not receipt.get("files")):
        raise ValueError("Candidate model receipt does not match reviewed checkpoint")
    snapshot = Path(receipt["snapshot"]).absolute()
    files = receipt["files"]
    expected_paths = {str(path) for path in snapshot.iterdir() if path.is_file()}
    if len(files) != len(expected_paths) or {item.get("path") for item in files} != expected_paths:
        raise ValueError("Candidate model receipt has incomplete file coverage")
    decision = pipeline.read_json(args.output / "candidate_model_decision.json", {})
    if decision.get("receipt_sha256") and decision["receipt_sha256"] != file_digest(args.candidate_model_receipt):
        raise ValueError("Pinned candidate model receipt changed")
    for item in files:
        path = Path(item["path"])
        if not path.is_absolute() or not path.is_relative_to(snapshot) or ".." in path.parts:
            raise ValueError("Candidate receipt path escapes its snapshot")
        current = signature(path)
        if any(item.get(key) != value for key, value in current.items()):
            raise ValueError("Candidate checkpoint changed after full integrity verification")
        if (item.get("actual_hash") != item.get("expected_hash")
                or item.get("actual_hash") != Path(current["resolved"]).name):
            raise ValueError("Candidate checkpoint lacks matching content hash evidence")
    return {"receipt_sha256": file_digest(args.candidate_model_receipt), "model": CPGRPO,
            "revision": args.cpgrpo_revision, "files_verified": len(receipt["files"])}


def build_campaign(args, name, *, manifest, backends, model=INSTRUCT, revision=INSTRUCT_REVISION, steps=64, split="dev",
                   compact_queries=True, rectangular_attention=True):
    output = args.output / name
    argv = ["functions", "--output", str(output), "--data-root", str(args.data_root), "--gpus", args.gpus,
        "--backends", ",".join(backends), "--suites", ",".join(SUITES), "--selection-manifest", str(manifest),
        "--tokens", "512", "--standard-steps", str(steps), "--block-size", "512", "--dard-steps", "64",
        "--tau-c", ".5", "--tau-u", ".8", "--transfer-limit", "0", "--repairs", "0", "--repeats", "1",
        "--seed", "42", "--quant", "bf16", "--diagnostics", "summary", "--split", split, "--fail-fast", "--execute"]
    options = four_gpu.parser().parse_args(argv)
    plan, tasks = four_gpu.make_plan(options, four_gpu.resolve_gpus(args.gpus))
    for worker in plan["workers"]:
        command = worker["command"]
        command[command.index("--revision")+1] = revision
        command.extend(["--model", model])
        if not compact_queries:
            command.append("--no-compact-queries")
        if not rectangular_attention:
            command.append("--no-rectangular-attention")
        from main import cli
        cli(command[2:])
    plan["optimization_candidate"] = {"model": model, "revision": revision, "native_steps": steps,
        "partition_sha256": file_digest(manifest), "final_test": split == "test",
        "compact_queries": compact_queries, "rectangular_attention": rectangular_attention}
    return output, plan, tasks


def run_campaign(args, state, name, *, deadline, **options):
    pipeline.verify_plan_sources(pipeline.read_json(args.output / "optimization_plan.json"), repo=ROOT)
    event(args, state, name)
    output, plan, tasks = build_campaign(args, name, **options)
    four_gpu.save_plan(output, plan)
    with (output / "campaign.lock").open("a") as lock:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        summary = pipeline.read_json(output / "campaign_summary.json", {})
        if summary.get("coverage_complete") and summary.get("groups") and all(
                group.get("complete_evaluated_cohort") is True for group in summary["groups"].values()):
            return read_rows(output, plan)
        with bounded(deadline), four_gpu.owned_descendant_cleanup():
            success = four_gpu.execute(plan, tasks, output, fail_fast=True)
        coverage = four_gpu.audit(plan, output)
        if not success or not coverage["coverage_complete"]:
            raise RuntimeError("Candidate campaign failed or has incomplete coverage: " + name)
        return read_rows(output, plan)


def read_rows(output, plan):
    latest = {}
    expected = set()
    for worker in plan["workers"]:
        for task in worker["task_keys"]:
            for backend in plan["backends"]:
                expected.add((*task, backend, 0, 42))
        path = Path(worker["output"]) / "case_results.jsonl"
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["dataset"], row["language"], row["task_id"], row["decoder"], row["repeat"], row["seed"])
            latest[key] = row
    if set(latest) != expected:
        raise ValueError("Candidate results do not exactly cover declared tasks and backends")
    validate_measurements(latest.values())
    return list(latest.values())


def validate_measurements(rows):
    for row in rows:
        seconds = row.get("generation_wall_s")
        if (row.get("case_completed") is not True or row.get("measurement_complete") is not True
                or type(row.get("final_pass")) is not bool or row.get("failure_kind")
                or not isinstance(seconds, (int, float)) or isinstance(seconds, bool)
                or not math.isfinite(seconds) or seconds <= 0):
            raise ValueError("Candidate contains incomplete or infrastructure-failed measurements")


def legacy_development(args, manifest):
    """Read only frozen dev identities after baseline completion; final rows never enter selection."""
    if file_digest(args.output / "development_selection.json") != pipeline.read_json(args.output / "partition_receipt.json")["development_manifest_sha256"]:
        raise ValueError("Development partition changed before legacy comparison")
    wanted = {tuple(key) for key in manifest["task_keys"]}
    plan_path = args.baseline_output / "functions/campaign.json"
    plan = json.loads(plan_path.read_text())
    rows = {}
    for worker in plan["workers"]:
        output = Path(worker["output"])
        settings = json.loads((output / "config.json").read_text())["settings"]
        for key, expected in (("model", INSTRUCT), ("revision", INSTRUCT_REVISION),
                              ("tokens", 512), ("standard_steps", 64), ("repairs", 0),
                              ("temperature", 0), ("top_p", 1)):
            if settings.get(key) != expected:
                raise ValueError("Legacy development reference has different pinned protocol: " + key)
        with (output / "case_results.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                identity = (row["dataset"], row["language"], row["task_id"])
                if identity in wanted and row["decoder"] in ("ar", "dard", "standard"):
                    rows[(*identity, row["decoder"], row["repeat"], row["seed"])] = row
    expected = {(*key, backend, 0, 42) for key in wanted for backend in ("ar", "dard", "standard")}
    if set(rows) != expected:
        raise ValueError("Legacy development reference lacks exact complete cohort")
    validate_measurements(rows.values())
    record = {"schema_version": 1, "read_unix": time.time(), "partition_frozen_unix": pipeline.read_json(args.output / "partition_receipt.json")["created_unix"],
        "baseline_plan_sha256": file_digest(plan_path), "development_manifest_sha256": file_digest(args.output / "development_selection.json"),
        "final_test_outcomes_used": False, "cross_run_timing_note": "Exploratory reference from the preserved baseline; GPUs may assign these dev identities to different shards. Final selected protocol is measured again on all three methods.",
        "rows": list(rows.values())}
    path = args.output / "legacy_development_reference.json"
    previous = pipeline.read_json(path)
    if previous:
        record["read_unix"] = previous["read_unix"]
    freeze_json(path, record)
    return {"model": INSTRUCT, "revision": INSTRUCT_REVISION, "compact_queries": False, "rectangular_attention": False,
        "dard": [r for r in rows.values() if r["decoder"] == "dard"],
        "standard": [r for r in rows.values() if r["decoder"] == "standard"]}


def describe(rows):
    if not rows:
        raise ValueError("Empty candidate results")
    times = sorted(row["generation_wall_s"] for row in rows)
    def language_stats(language):
        selected = [row for row in rows if row["language"] == language]
        values = sorted(row["generation_wall_s"] for row in selected)
        return {"cases": len(selected), "passed": sum(row["final_pass"] for row in selected),
            "pass_rate": sum(row["final_pass"] for row in selected)/len(selected) if selected else None,
            "mean_generation_seconds": sum(values)/len(values) if values else None,
            "p95_generation_seconds": values[math.ceil(.95*len(values))-1] if values else None}
    return {"cases": len(rows), "passed": sum(row["final_pass"] for row in rows),
        "pass_rate": sum(row["final_pass"] for row in rows)/len(rows),
        "mean_generation_seconds": sum(times)/len(times),
        "p95_generation_seconds": times[math.ceil(.95*len(times))-1],
        "by_language": {language: language_stats(language) for language in LANGUAGES},
        "failure_statuses": dict(Counter(row.get("final_status", "unknown") for row in rows if not row["final_pass"]))}


def compare_rows(reference, candidate):
    def keyed(rows):
        return {(r["dataset"], r["language"], r["task_id"], r["repeat"], r["seed"]): r for r in rows}
    left, right = keyed(reference), keyed(candidate)
    if len(left) != len(reference) or len(right) != len(candidate) or set(left) != set(right):
        raise ValueError("Comparison requires identical unique complete task cohorts")
    return dict(Counter("both_pass" if left[k]["final_pass"] and right[k]["final_pass"] else
        "both_fail" if not left[k]["final_pass"] and not right[k]["final_pass"] else
        "regression" if left[k]["final_pass"] else "improvement" for k in left))


def choose(reference, candidates, *, margin=.03):
    ar = describe(reference)
    records = []
    for name, data in candidates.items():
        report = describe(data["dard"])
        paired = compare_rows(reference, data["dard"])
        native_paired = compare_rows(data["standard"], data["dard"])
        eligible = (report["pass_rate"] + margin + 1e-12 >= ar["pass_rate"]
                    and report["mean_generation_seconds"] < ar["mean_generation_seconds"])
        records.append({"name": name, "model": data["model"], "revision": data["revision"],
            "compact_queries": data.get("compact_queries", True), "rectangular_attention": data.get("rectangular_attention", True),
            "dard": report, "native64": describe(data["standard"]), "paired_vs_ar": paired,
            "paired_dard_vs_native64": native_paired, "goal_screen_passed": eligible,
            "native256": describe(data["quality"]) if data.get("quality") else None,
            "paired_dard_vs_native256": compare_rows(data["quality"], data["dard"]) if data.get("quality") else None,
            "paired_native64_vs_native256": compare_rows(data["quality"], data["standard"]) if data.get("quality") else None})
    if not records:
        raise ValueError("No complete candidate available")
    eligible = [r for r in records if r["goal_screen_passed"]]
    selected = min(eligible, key=lambda r: (r["dard"]["mean_generation_seconds"], -r["dard"]["pass_rate"], r["name"])) if eligible else min(
        records, key=lambda r: (-r["dard"]["pass_rate"], r["dard"]["mean_generation_seconds"], r["name"]))
    return {"schema_version": 1, "selected": selected["name"], "model": selected["model"],
        "revision": selected["revision"], "goal_screen_passed": selected["goal_screen_passed"],
        "compact_queries": selected["compact_queries"], "rectangular_attention": selected["rectangular_attention"],
        "selection_reason": "fastest_within_accuracy_margin" if eligible else "goal_unmet_best_development_accuracy_then_time",
        "accuracy_margin": margin, "screening_is_statistical_noninferiority": False,
        "ar": ar, "candidates": records, "final_standard_steps": 64,
        "native256_role": "quality diagnostic only; no additional final operating point",
        "final_validation_required": True}


def final_result(rows, goal_report, *, margin=.03):
    """Describe the fixed full test; never select cases or methods from its outcomes."""
    groups = {backend: [r for r in rows if r["decoder"] == backend] for backend in ("ar", "dard", "standard")}
    summaries = {backend: describe(selected) for backend, selected in groups.items()}
    outcomes = compare_rows(groups["ar"], groups["dard"])
    compare_rows(groups["standard"], groups["dard"])
    delta = summaries["dard"]["pass_rate"] - summaries["ar"]["pass_rate"]
    speed = summaries["ar"]["mean_generation_seconds"] / summaries["dard"]["mean_generation_seconds"]
    identity = lambda r: (r["dataset"], r["language"], r["task_id"], r["repeat"], r["seed"])
    lengths = {identity(r): r.get("visible_output_tokens") for r in groups["ar"]}
    bins = {}
    for label, low, high in (("0_to_64", 0, 64), ("65_to_128", 65, 128), ("129_plus", 129, math.inf)):
        selected_keys = {key for key, value in lengths.items() if isinstance(value, (int, float)) and low <= value <= high}
        if selected_keys:
            bins[label] = {backend: describe([r for r in selected if identity(r) in selected_keys]) for backend, selected in groups.items()}
    controlled = goal_report.get("comparisons", {}).get("dard_vs_ar/all", {})
    interval = controlled.get("pass_at_1_delta_source_cluster_bootstrap_95ci")
    supported = bool(controlled.get("complete_known_controlled_cohort") is True and interval
                     and len(interval) == 2 and interval[0] >= -margin)
    return {"schema_version": 1, "accuracy_margin": margin, "descriptive_accuracy_delta_dard_minus_ar": delta,
        "generation_speedup_ar_over_dard": speed,
        "observed_full_cohort_criteria_met": delta + 1e-12 >= -margin and speed > 1,
        "statistical_noninferiority_established": False, "goal_claim": "Descriptive full-cohort criteria only; inspect source-cluster confidence intervals and language groups before making a statistical claim.",
        "accuracy_margin_supported_by_source_bootstrap": supported,
        "controlled_comparison_eligible": controlled.get("complete_known_controlled_cohort") is True,
        "groups": summaries, "paired_dard_vs_ar": outcomes,
        "paired_dard_vs_native64": compare_rows(groups["standard"], groups["dard"]),
        "length_bucket_definition": "Same task identities for every backend, anchored to AR visible output token count; includes failed answers.",
        "length_buckets": bins, "length_bucket_missing_reference_cases": sum(value is None for value in lengths.values()),
        "controlled_source_cluster_comparisons": {key: value for key, value in goal_report.get("comparisons", {}).items() if key.startswith("dard_vs_ar/")},
        "final_test_was_not_used_for_selection": True}


def optional_campaign(args, state, name, *, deadline, estimated_seconds, **options):
    """A development diagnostic timeout must not discard complete candidates."""
    record_path = args.output / name / "optional_stage.json"
    previous = pipeline.read_json(record_path, {})
    if previous.get("status") in ("skipped", "timed_out"):
        return None
    if time.time() + estimated_seconds >= deadline:
        pipeline.write_json(record_path, {"status": "skipped", "reason": "Preserve development/final deadline reserve",
            "estimated_seconds": estimated_seconds, "deadline_unix": deadline, "decided_unix": time.time()})
        return None
    try:
        result = run_campaign(args, state, name, deadline=deadline, **options)
    except TimeoutError as exc:
        failure = pipeline.failure_event(args.output, state["invocation_id"], name + "_timeout", 124,
            error_type=type(exc).__name__, log_path=str(args.output / name))
        pipeline.write_json(record_path, {"status": "timed_out", "failure_event": failure,
            "candidate_excluded": True, "ended_unix": time.time()})
        pipeline.send_failure(args.output, failure)
        return None
    pipeline.write_json(record_path, {"status": "completed", "ended_unix": time.time()})
    return result


def _run(args, state, plan):
    pipeline.verify_plan_sources(plan, repo=ROOT)
    verify_receipt(args.data_root, plan["prepared_model_receipt"], plan["prepared_model_receipt_sha256"])
    wait_baseline(args, state)
    pipeline.verify_plan_sources(plan, repo=ROOT)
    event(args, state, "numerical_gpu_validation")
    validate_gate(args)
    marker = args.output / "development_started.json"
    began = pipeline.read_json(marker)
    if not began:
        baseline_timing = pipeline.read_json(args.baseline_output / "functions/execution_status.json", {})
        observed = baseline_timing.get("invocation_wall_s")
        reserve = max(1800, 1.5*observed + 1800) if isinstance(observed, (int, float)) and math.isfinite(observed) and observed > 0 else 6*3600
        began = {"started_unix": time.time(), "final_and_analysis_reserve_seconds": reserve,
            "reserve_basis": "1.5 times completed baseline invocation wall time plus 1800 seconds; six hours if unavailable",
            "deadline_unix": min(args.deadline_unix-reserve, time.time()+args.development_budget_seconds)}
        pipeline.write_json(marker, began)
    dev_deadline = began["deadline_unix"]
    dev = args.output / "development_selection.json"
    final = args.output / "evaluation_selection.json"
    decision_path = args.output / "selection_decision.json"
    decision = pipeline.read_json(decision_path)
    if not decision:
        instruct = run_campaign(args, state, "development/instruct", deadline=dev_deadline,
                                manifest=dev, backends=["ar", "dard", "standard"])
        ar = [r for r in instruct if r["decoder"] == "ar"]
        candidates = {"instruct": {"model": INSTRUCT, "revision": INSTRUCT_REVISION,
            "dard": [r for r in instruct if r["decoder"] == "dard"],
            "standard": [r for r in instruct if r["decoder"] == "standard"]}}
        legacy = legacy_development(args, json.loads(dev.read_text()))
        candidates["legacy_instruct"] = legacy
        if candidate_available(args, state, dev_deadline):
            cp = optional_campaign(args, state, "development/cpgrpo", deadline=dev_deadline, estimated_seconds=1800,
                manifest=dev, backends=["dard", "standard"], model=CPGRPO, revision=args.cpgrpo_revision)
            if cp is not None:
                candidates["cpgrpo"] = {"model": CPGRPO, "revision": args.cpgrpo_revision,
                    "dard": [r for r in cp if r["decoder"] == "dard"],
                    "standard": [r for r in cp if r["decoder"] == "standard"]}
        for name in ("instruct", "cpgrpo"):
            if name not in candidates:
                continue
            candidate = candidates[name]
            # Four times as many native steps, 2x safety, four workers, plus
            # model load and grading reserve. This is an estimate, never a guarantee.
            estimate = 2 * sum(r["generation_wall_s"] for r in candidate["standard"]) + 120
            candidate["quality"] = optional_campaign(args, state, "development/"+name+"_quality",
                deadline=dev_deadline, estimated_seconds=estimate, manifest=dev, backends=["standard"],
                model=candidate["model"], revision=candidate["revision"], steps=256)
        legacy["quality"] = candidates["instruct"].get("quality")
        decision = choose(ar, candidates, margin=args.accuracy_margin)
        decision["optimized_instruct_vs_legacy_development"] = {
            "paired_outcomes": compare_rows(legacy["dard"], candidates["instruct"]["dard"]),
            "generation_speedup_legacy_over_optimized": describe(legacy["dard"])["mean_generation_seconds"] / describe(candidates["instruct"]["dard"])["mean_generation_seconds"],
            "legacy_reference_path": str(args.output / "legacy_development_reference.json")}
        decision.update(decided_unix=time.time(), development_manifest_sha256=file_digest(dev),
                        evaluation_manifest_sha256=file_digest(final), optimization_plan_sha256=file_digest(args.output/"optimization_plan.json"))
        freeze_json(decision_path, decision)
    else:
        if (decision.get("development_manifest_sha256") != file_digest(dev)
                or decision.get("evaluation_manifest_sha256") != file_digest(final)
                or decision.get("optimization_plan_sha256") != file_digest(args.output/"optimization_plan.json")):
            raise ValueError("Selection decision binding changed")
    event(args, state, "final_selection_frozen", selected=decision["selected"], goal_screen_passed=decision["goal_screen_passed"])
    if decision["model"] == CPGRPO:
        verify_candidate_receipt(args)
    final_rows = run_campaign(args, state, "functions", deadline=args.deadline_unix-300, manifest=final,
        backends=["ar", "dard", "standard"], model=decision["model"], revision=decision["revision"], split="test",
        compact_queries=decision["compact_queries"], rectangular_attention=decision["rectangular_attention"])
    event(args, state, "final_analysis")
    goal_report = write_goal_report(args.output / "functions")
    pipeline.write_json(args.output / "optimization_result.json", final_result(final_rows, goal_report, margin=args.accuracy_margin))
    event(args, state, "completed", status="completed", ended_unix=time.time())
    return 0


def notification_only(args, state):
    """After three failed attempts, retry only durable messages, never GPU work."""
    import experiment_notifications as mail
    event(args, state, "notification_retry_only", status="blocked", reason="Three infrastructure attempts failed; experiment restart disabled")
    while time.time() < args.deadline_unix:
        fallback = pipeline.read_json(args.output / "notification_fallback.json", {})
        if fallback.get("pending"):
            receipt = pipeline.send_failure(args.output, fallback["event"])
            pipeline.write_json(args.output / "notification_fallback.json", {"event": fallback["event"], **receipt})
        flushed = mail.flush_pending(args.output, max_records=20, deadline_seconds=45)
        pipeline.write_json(args.output / "notification_monitor.json", {"updated_unix": time.time(), **flushed})
        if (flushed.get("pending_remaining") == 0 and flushed.get("remaining_scan_complete", True)
                and not flushed.get("errors") and not pipeline.read_json(args.output / "notification_fallback.json", {}).get("pending")):
            return 0
        time.sleep(min(15, max(.01, args.deadline_unix-time.time())))
    return 0


def post_stop(args):
    """systemd ExecStopPost hook reports loss even after SIGKILL or OOM."""
    import experiment_notifications as mail
    state = pipeline.read_json(args.output / "optimization_state.json", {})
    if not state and os.environ.get("SERVICE_RESULT", "success") != "success":
        # The process may be killed while acquiring/fingerprinting its first
        # plan, before the ordinary running state has been published.
        state = {"invocation_id": "startup-loss-" + uuid.uuid4().hex,
                 "status": "running", "failure_count": 0, "stage": "startup"}
    if state.get("status") == "running":
        failure = pipeline.failure_event(args.output, state["invocation_id"], "optimization_process_lost", 1,
            error_type="OptimizationProcessLost", log_path=str(args.output/"service.log"))
        pipeline.write_json(args.output/"optimization_failure.json", {"event": failure, "service_result": os.environ.get("SERVICE_RESULT", "unknown")})
        event(args, state, "optimization_process_lost", status="failed", failure_count=state.get("failure_count", 0)+1)
        pipeline.send_failure(args.output, failure)
    mail.flush_pending(args.output, max_records=20, deadline_seconds=45)
    return 0


def run(args):
    import fcntl
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/"optimization.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args._lock_acquired = True
        os.environ.pop("HYBRID_DARD_DISABLE_EMAIL", None)
        if args.post_stop:
            return post_stop(args)
        previous = pipeline.read_json(args.output/"optimization_state.json")
        if args.execute and previous and previous.get("failure_count", 0) >= 3:
            return notification_only(args, previous)
        plan = prepare(args)
        if not args.execute:
            print(json.dumps({"status": "prepared", "output": str(args.output), "gpu_work_started": False}))
            return 0
        state = {"schema_version": 1, "invocation_id": uuid.uuid4().hex, "status": "running",
                 "stage": "starting", "started_unix": time.time(), "deadline_unix": args.deadline_unix,
                 "failure_count": previous.get("failure_count", 0) if previous else 0}
        if previous and previous.get("status") == "completed":
            return 0
        if previous and previous.get("status") == "running":
            lost = pipeline.failure_event(args.output, previous["invocation_id"], "optimization_restarted", 1,
                error_type="OptimizationProcessLost", log_path=str(args.output/"service.log"))
            pipeline.send_failure(args.output, lost)
            state["failure_count"] += 1
        if state["failure_count"] >= 3:
            return notification_only(args, state)
        handlers = pipeline.install_handlers()
        try:
            with bounded(args.deadline_unix-60):
                return _run(args, state, plan)
        except BaseException as exc:
            code = 124 if isinstance(exc, TimeoutError) else 1
            failure = pipeline.failure_event(args.output, state["invocation_id"], state["stage"], code,
                error_type=type(exc).__name__, log_path=str(args.output/"service.log"))
            pipeline.write_json(args.output/"optimization_failure.json", {"event": failure, "error": str(exc), "traceback": traceback.format_exc()})
            event(args, state, state["stage"], status="failed", exit_code=code, ended_unix=time.time(), failure_count=state["failure_count"]+1)
            pipeline.send_failure(args.output, failure)
            args._failure_notified = True
            raise
        finally:
            pipeline.restore_handlers(handlers)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline-output", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("/data/hybrid-dard"))
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--deadline-unix", type=float, required=True)
    p.add_argument("--development-sources-per-family", type=int, default=8)
    p.add_argument("--development-budget-seconds", type=float, default=4*3600)
    p.add_argument("--accuracy-margin", type=float, default=.03)
    p.add_argument("--cpgrpo-revision", choices=(CPGRPO_REVISION,))
    p.add_argument("--candidate-model-receipt", type=Path)
    p.add_argument("--optional-model-wait-seconds", type=float, default=3600)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--post-stop", action="store_true", help="systemd ExecStopPost loss notification; never starts experiments")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ("output", "baseline_output", "data_root", "candidate_model_receipt"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if (not math.isfinite(args.deadline_unix) or args.deadline_unix <= 0
            or not math.isfinite(args.development_budget_seconds) or args.development_budget_seconds <= 0
            or not math.isfinite(args.optional_model_wait_seconds) or args.optional_model_wait_seconds < 0):
        raise ValueError("Deadlines and time budgets must be finite and valid")
    try:
        return run(args)
    except BaseException as exc:
        # Configuration and lock acquisition can fail before run() enters its
        # inner handler. Keep these failures visible to the same mail outbox.
        if not getattr(args, "_failure_notified", False) and not args.post_stop:
            args.output.mkdir(parents=True, exist_ok=True)
            event_id = uuid.uuid4().hex
            failure = pipeline.failure_event(args.output, event_id, "optimization_startup", 1,
                error_type=type(exc).__name__, log_path=str(args.output / "service.log"))
            if getattr(args, "_lock_acquired", False):
                previous = pipeline.read_json(args.output / "optimization_state.json", {})
                state = {"invocation_id": event_id, "status": "failed", "stage": "optimization_startup",
                         "failure_count": previous.get("failure_count", 0)+1,
                         "deadline_unix": args.deadline_unix, "ended_unix": time.time()}
                pipeline.write_json(args.output / "optimization_state.json", state)
                pipeline.write_json(args.output / "optimization_failure.json", {"event": failure, "error": str(exc), "traceback": traceback.format_exc()})
            pipeline.send_failure(args.output, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
