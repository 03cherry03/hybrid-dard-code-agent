"""CPU analysis of frozen validation evaluations, with source-cluster uncertainty.

Functional correctness and completion certification are separate measurements.
Selection is exploratory: searching a grid and reusing validation are not an
independent noninferiority test. Never call this module a global optimizer.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import html
import json
import math
from pathlib import Path
import statistics
import time

from iterative_tuning import digest, file_sha, task_id, write_json

SCHEMA_VERSION = 1
INFRA_STATUSES = {"infra_error", "infrastructure_error", "not_tested", "environment_error"}
GENERATION_FAILURES = {"ForwardBudgetExceeded", "GenerationDeadlineExceeded"}
STATISTICAL_LIMITATION = (
    "Exploratory validation selection, not an independent confirmatory test. "
    "Confidence intervals are paired source-cluster percentile bootstrap intervals; "
    "all languages and evaluation seeds of a source remain together. Intervals are "
    "not multiplicity- or selection-adjusted. No global optimum or statistical "
    "noninferiority is established. Single-seed results do not quantify sampling-seed variation.")


def _read(path):
    return json.loads(Path(path).read_text())


def _finite(value, *, nonnegative=True):
    return (type(value) in (int, float) and math.isfinite(value)
            and (not nonnegative or value >= 0))


def _integer(value):
    return type(value) is int and value >= 0


def _adapter_hashes(config):
    adapter = config.get("adapter")
    if adapter is None:
        return None
    hashes = adapter.get("sha256")
    required = {"adapter_config.json", "adapter_model.safetensors"}
    if (not isinstance(hashes, dict) or not required <= hashes.keys()
            or any(not isinstance(v, str) or len(v) != 64
                   or any(c not in "0123456789abcdef" for c in v) for v in hashes.values())):
        raise ValueError("Adapter identity requires recorded file SHA256 hashes")
    return dict(sorted(hashes.items()))


def _normalized_config(config):
    result = {key: value for key, value in config.items() if key != "id"}
    result["adapter"] = _adapter_hashes(config)
    return result


def comparable_settings(config, taskset_sha256, seeds):
    """Strict cohort identity; decoder schedule is intentionally varied within it."""
    for key in ("model", "revision", "prefix", "tokens", "dard"):
        if key not in config:
            raise ValueError("Candidate lacks comparable-setting identity: " + key)
    return {"model": config["model"], "revision": config["revision"],
            "adapter_sha256": _adapter_hashes(config), "prefix": config["prefix"],
            "temperature": config["dard"]["temperature"], "top_p": config["dard"]["top_p"],
            "tokens": config["tokens"], "taskset_sha256": taskset_sha256,
            "seeds": sorted(seeds)}


def _diagnostic_fields(row, case_path):
    """Read metadata only. No model/tokenizer loads and no parsing generated code."""
    generation = {}
    receipt_path = row.get("generation_receipt")
    if receipt_path:
        path = Path(receipt_path)
        if not path.is_absolute():
            path = case_path.parent / path
        if row.get("generation_receipt_sha256") != file_sha(path):
            raise ValueError("Generation receipt SHA256 mismatch")
        receipt = _read(path)
        if any(receipt.get(key) != row.get(key) for key in
               ("candidate_id", "task_key", "task_sha256", "config_sha256")):
            raise ValueError("Generation receipt identity mismatch")
        generation = receipt.get("generation") or {}
    def field(name, legacy=None):
        if name in row:
            return row[name]
        return generation.get(name, generation.get(legacy) if legacy else None)
    incomplete = field("incomplete")
    certified = field("verification_certified")
    unresolved = field("unresolved_mask_tokens", "unfilled_mask_tokens")
    if incomplete is not None and type(incomplete) is not bool:
        raise ValueError("Invalid incomplete-output flag")
    if certified is not None and type(certified) is not bool:
        raise ValueError("Invalid verification-certification flag")
    if unresolved is not None and not _integer(unresolved):
        raise ValueError("Invalid unresolved MASK count")
    return {"incomplete": incomplete, "verification_certified": certified,
            "unresolved_mask_tokens": unresolved,
            "length_limit_reached": generation.get("stop_reason") == "length" if generation else None}


def load_evaluations(phase_dirs, *, candidate_ids=None):
    """Load raw cases against immutable task/config identities; reject holdout input.

    Same settings across distinct seeds form one candidate. Repeated phases of
    the same candidate/seed invalidate that candidate rather than count twice.
    Missing or malformed cases and infrastructure errors remain visible and
    disqualify selection. Output incompleteness alone does not change pass/fail.
    """
    candidates = {}
    phase_receipts = []
    requested = set(candidate_ids) if candidate_ids is not None else None
    if requested is not None and (not requested or any(not isinstance(cid, str) for cid in requested)):
        raise ValueError("candidate_ids must be a nonempty collection of IDs")
    found_requested = set()
    phases = [Path(path).resolve() for path in phase_dirs]
    if not phases or len(phases) != len(set(phases)):
        raise ValueError("Provide distinct nonempty evaluation phase paths")
    for phase in sorted(phases):
        plan_path = phase / "evaluation_plan.json"
        plan = _read(plan_path)
        if plan.get("split") != "validation":
            raise ValueError("Only validation input is permitted for selection; holdout is forbidden")
        seed = plan.get("seed")
        if not _integer(seed):
            raise ValueError("Evaluation plan requires a nonnegative integer seed")
        tasks_path = Path(plan["tasks_path"])
        if not tasks_path.is_absolute():
            tasks_path = phase / tasks_path
        if file_sha(tasks_path) != plan["tasks_sha256"]:
            raise ValueError("Evaluation task file changed")
        tasks = [_read_line(line) for line in tasks_path.read_text().splitlines() if line.strip()]
        expected = {task_id(task): task for task in tasks}
        if not expected or len(expected) != len(tasks):
            raise ValueError("Task manifest must have unique nonempty identities")
        source_languages = [(task["source_id"], task["language"]) for task in tasks]
        if len(set(source_languages)) != len(source_languages):
            raise ValueError("Task manifest duplicates a source/language identity")
        taskset_sha = digest(sorted((key, digest(task)) for key, task in expected.items()))
        grading_workers = plan.get("grading_workers", 2)
        grading_queue_size = plan.get("grading_queue_size", 4)
        if type(grading_workers) is not int or not 1 <= grading_workers <= 8:
            raise ValueError("Invalid evaluation grading_workers")
        if type(grading_queue_size) is not int or not grading_workers <= grading_queue_size <= 128:
            raise ValueError("Invalid evaluation grading_queue_size")
        protocol = {"extraction_version": plan.get("extraction_version", "legacy"),
                    "generation_timeout": plan.get("generation_timeout"),
                    "grading_timeout": plan.get("grading_timeout"),
                    "grading_workers": grading_workers,
                    "grading_queue_size": grading_queue_size,
                    "candidate_warmup": plan.get("candidate_warmup", False),
                    "counterbalanced_order": plan.get("counterbalanced_order", False)}
        phase_receipts.append({"path": str(phase), "plan_sha256": file_sha(plan_path),
                               "tasks_file_sha256": plan["tasks_sha256"],
                               "taskset_sha256": taskset_sha, "seed": seed, "tasks": len(tasks)})
        phase_ids = set()
        for config in plan["candidates"]:
            cid = config["id"]
            if (not isinstance(cid, str) or not cid or Path(cid).name != cid
                    or cid in {".", ".."} or cid in phase_ids):
                raise ValueError("Invalid or duplicate candidate ID in plan")
            phase_ids.add(cid)
            if requested is not None and cid not in requested:
                continue
            found_requested.add(cid)
            config_identity = _normalized_config(config)
            comparable_settings(config, taskset_sha, [seed])
            key = "candidate-" + digest([config_identity, taskset_sha, protocol])[:24]
            record = candidates.setdefault(key, {"analysis_id": key, "config": config,
                "config_identity": config_identity, "candidate_ids": set(), "seeds": set(),
                "taskset_sha256": taskset_sha, "expected_per_seed": len(tasks),
                "evaluation_protocol": protocol,
                "expected_languages": dict(Counter(task["language"] for task in tasks)),
                "source_ids": {task["source_id"] for task in tasks}, "rows": {},
                "issues": [], "phase_paths": []})
            record["candidate_ids"].add(cid)
            record["phase_paths"].append(str(phase))
            if seed in record["seeds"]:
                record["issues"].append({"type": "duplicate_candidate_seed_phase", "phase": str(phase), "seed": seed})
                continue
            record["seeds"].add(seed)
            seen = set()
            for case_path in sorted(phase.glob(f"shard-*/cases/{cid}/*.json")):
                try:
                    row = _read(case_path)
                    task_key = row["task_key"]
                    if task_key in seen:
                        raise ValueError("Duplicate case identity")
                    seen.add(task_key)
                    task = expected[task_key]
                    if (row.get("split") != "validation" or row.get("candidate_id") != cid
                            or row.get("config_sha256") != digest(config)
                            or row.get("task_sha256") != digest(task)
                            or row.get("source_id") != task["source_id"]
                            or row.get("language") != task["language"]
                            or ("measurement_seed" in row and row["measurement_seed"] != seed)):
                        raise ValueError("Case task/config/split identity mismatch")
                    if (type(row.get("passed")) is not bool
                            or type(row.get("infrastructure_error")) is not bool
                            or not _finite(row.get("generation_seconds"))
                            or not _integer(row.get("forward_calls"))
                            or not _integer(row.get("forced_tokens"))):
                        raise ValueError("Invalid correctness or all-case latency/forward metadata")
                    if (row["passed"] != (row.get("status") == "passed")):
                        raise ValueError("Contradictory pass/status record")
                    normalized = {name: row[name] for name in (
                        "task_key", "task_sha256", "source_id", "language", "passed", "status",
                        "generation_seconds", "forward_calls", "forced_tokens", "infrastructure_error")}
                    normalized.update(evaluation_seed=seed, **_diagnostic_fields(row, case_path))
                    record["rows"][(seed, task_key)] = normalized
                    if row["infrastructure_error"] or row["status"] in INFRA_STATUSES:
                        record["issues"].append({"type": "infrastructure_error", "case": str(case_path)})
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    record["issues"].append({"type": "invalid_case", "case": str(case_path),
                                             "error_type": type(exc).__name__, "detail": str(exc)[:200]})
            missing = set(expected) - {key[1] for key in record["rows"] if key[0] == seed}
            if missing:
                record["issues"].append({"type": "missing_cases", "seed": seed, "count": len(missing)})
    if requested is not None and requested - found_requested:
        raise ValueError("Requested candidate IDs absent from input plans: " + ", ".join(sorted(requested-found_requested)))
    return list(candidates.values()), phase_receipts


def _read_line(line):
    return json.loads(line)


def _quantile(values, probability):
    values = sorted(values)
    if not values:
        return None
    index = (len(values)-1) * probability
    lower, upper = math.floor(index), math.ceil(index)
    return values[lower] + (values[upper]-values[lower]) * (index-lower)


def summarize_candidate(record):
    rows = list(record["rows"].values())
    expected = record["expected_per_seed"] * len(record["seeds"])
    incomplete = [row["incomplete"] for row in rows if row["incomplete"] is not None]
    certified = [row["verification_certified"] for row in rows if row["verification_certified"] is not None]
    unresolved = [row["unresolved_mask_tokens"] for row in rows if row["unresolved_mask_tokens"] is not None]
    settings = comparable_settings(record["config"], record["taskset_sha256"], record["seeds"])
    settings["evaluation_protocol"] = record["evaluation_protocol"]
    languages = {}
    for language, per_seed in sorted(record["expected_languages"].items()):
        subset = [row for row in rows if row["language"] == language]
        count = per_seed*len(record["seeds"])
        passed = sum(row["passed"] for row in subset)
        languages[language] = {"cases": count, "observed_cases": len(subset),
            "passed": passed, "pass_rate": passed/count,
            "mean_generation_seconds": statistics.mean(row["generation_seconds"] for row in subset) if subset else None}
    qualified = sum(row["passed"] and row["incomplete"] is False
                    and row["verification_certified"] is True
                    and row["unresolved_mask_tokens"] == 0 for row in rows)
    return {"analysis_id": record["analysis_id"], "candidate_id": min(record["candidate_ids"]),
        "candidate_ids": sorted(record["candidate_ids"]), "candidate": record["config"],
        "cohort_id": "cohort-" + digest(settings)[:24], "comparable_settings": settings,
        "phase_paths": record["phase_paths"], "seeds": sorted(record["seeds"]),
        "source_clusters": len(record["source_ids"]), "expected_cases": expected,
        "observed_cases": len(rows), "missing_or_invalid_cases": expected-len(rows),
        "valid_for_selection": not record["issues"] and len(rows) == expected,
        "issues": record["issues"], "passed": sum(row["passed"] for row in rows),
        "pass_rate": sum(row["passed"] for row in rows)/expected,
        "macro_language_pass_rate": statistics.mean(item["pass_rate"] for item in languages.values()),
        "macro_language_count": len(languages),
        "mean_generation_seconds": statistics.mean(row["generation_seconds"] for row in rows) if rows else None,
        "p95_generation_seconds": _quantile([row["generation_seconds"] for row in rows], .95),
        "mean_actual_forward_calls": statistics.mean(row["forward_calls"] for row in rows) if rows else None,
        "p95_actual_forward_calls": _quantile([row["forward_calls"] for row in rows], .95),
        "forced_case_count": sum(row["forced_tokens"] > 0 for row in rows),
        "forced_case_rate": sum(row["forced_tokens"] > 0 for row in rows)/expected,
        "mean_forced_tokens": statistics.mean(row["forced_tokens"] for row in rows) if rows else None,
        "generation_failure_count": sum(row["status"] in GENERATION_FAILURES for row in rows),
        "infrastructure_error_count": sum(row["infrastructure_error"] or row["status"] in INFRA_STATUSES for row in rows),
        "incomplete_output_count": sum(incomplete), "incomplete_output_known_cases": len(incomplete),
        "incomplete_output_rate": sum(incomplete)/len(incomplete) if incomplete else None,
        "verification_certified_count": sum(certified), "verification_certified_known_cases": len(certified),
        "unresolved_mask_tokens": sum(unresolved), "unresolved_mask_known_cases": len(unresolved),
        "completion_qualified_passed": qualified, "completion_qualified_pass_rate": qualified/expected,
        "completion_qualified_definition": "passed AND incomplete=false AND verification_certified=true AND unresolved_mask_tokens=0; unknown metadata is not certified",
        "by_language": languages,
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "latency_scope": "all observed cases, including incorrect and generation failures; incomplete evaluations cannot be selected"}


def paired_source_bootstrap(reference_cases, candidate_cases, *, repetitions=10000,
                            seed=20261105, confidence=.95):
    """Paired micro-accuracy difference; sample source clusters, never language rows."""
    import numpy as np
    if not _integer(repetitions) or repetitions < 100 or not 0 < confidence < 1:
        raise ValueError("Require at least 100 bootstrap draws and 0 < confidence < 1")
    def indexed(rows):
        result = {(row["evaluation_seed"], row["task_key"]): row for row in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate paired seed/task identity")
        return result
    reference, candidate = indexed(reference_cases), indexed(candidate_cases)
    if not reference or reference.keys() != candidate.keys():
        raise ValueError("Paired comparison requires identical complete task/seed identities")
    clusters = defaultdict(lambda: [0, 0, 0., 0.])
    wins = losses = both = 0
    for key in sorted(reference):
        left, right = reference[key], candidate[key]
        if any(left[name] != right[name] for name in ("source_id", "language", "task_sha256")):
            raise ValueError("Paired source/task content mismatch")
        wins += bool(right["passed"] and not left["passed"])
        losses += bool(left["passed"] and not right["passed"])
        both += bool(left["passed"] and right["passed"])
        cluster = clusters[left["source_id"]]
        cluster[0] += int(right["passed"]) - int(left["passed"])
        cluster[1] += 1
        cluster[2] += left["generation_seconds"]
        cluster[3] += right["generation_seconds"]
    values = np.asarray([clusters[key] for key in sorted(clusters)], dtype=float)
    rng = np.random.default_rng(seed)
    differences = []
    for offset in range(0, repetitions, 1000):
        indices = rng.integers(0, len(values), size=(min(1000, repetitions-offset), len(values)))
        sums = values[indices].sum(axis=1)
        differences.extend((sums[:, 0]/sums[:, 1]).tolist())
    tail = (1-confidence)/2
    return {"reference_only_correct": losses, "candidate_only_correct": wins,
        "both_correct": both, "both_incorrect": len(reference)-wins-losses-both,
        "cases": len(reference), "source_clusters": len(clusters),
        "evaluation_seeds": sorted({key[0] for key in reference}),
        "accuracy_difference": (wins-losses)/len(reference),
        "accuracy_difference_ci": np.quantile(differences, [tail, 1-tail]).tolist(),
        "confidence": confidence, "bootstrap_repetitions": repetitions, "bootstrap_seed": seed,
        "method": "paired source-cluster percentile bootstrap; within-source languages and seeds retained",
        "multiplicity_adjusted": False, "statistical_noninferiority_established": False}


def pareto_frontier(summaries):
    valid = [row for row in summaries if row["valid_for_selection"]]
    return [row["analysis_id"] for row in sorted(valid, key=lambda x: (x["mean_generation_seconds"], x["analysis_id"]))
            if not any(other["pass_rate"] >= row["pass_rate"]
                       and other["mean_generation_seconds"] <= row["mean_generation_seconds"]
                       and (other["pass_rate"] > row["pass_rate"]
                            or other["mean_generation_seconds"] < row["mean_generation_seconds"])
                       for other in valid)]


def analyze_evaluations(phase_dirs, *, output=None, bootstrap_repetitions=10000,
                        bootstrap_seed=20261105, quality_tolerance=.03,
                        latency_budgets=(1., 2., 5., 10.), write_svg=True,
                        candidate_ids=None):
    """Return an auditable validation-only report and optionally write JSON/CSV/SVG.

    Quality maximum is empirical within each strict cohort. A faster practical
    choice must clear the exploratory paired CI margin against that maximum.
    AR gates are separately labeled cross-model accuracy targets, never merged
    into a model-specific Pareto frontier.
    """
    if not _finite(quality_tolerance) or quality_tolerance > 1:
        raise ValueError("quality_tolerance must lie in [0,1]")
    budgets = sorted(set(latency_budgets))
    if any(not _finite(value) or value <= 0 for value in budgets):
        raise ValueError("Latency budgets must be finite positive seconds")
    if not _integer(bootstrap_seed) or not _integer(bootstrap_repetitions) or bootstrap_repetitions < 100:
        raise ValueError("Invalid bootstrap seed or repetition count")
    records, inputs = load_evaluations(phase_dirs, candidate_ids=candidate_ids)
    summaries = [summarize_candidate(record) for record in records]
    by_id = {record["analysis_id"]: record for record in records}
    groups = defaultdict(list)
    for summary in summaries:
        groups[summary["cohort_id"]].append(summary)
    comparisons = {}
    def compare(reference, candidate):
        key = reference["analysis_id"], candidate["analysis_id"]
        if key not in comparisons:
            comparisons[key] = {"reference_analysis_id": key[0], "candidate_analysis_id": key[1],
                **paired_source_bootstrap(list(by_id[key[0]]["rows"].values()),
                    list(by_id[key[1]]["rows"].values()), repetitions=bootstrap_repetitions,
                    seed=bootstrap_seed)}
        return comparisons[key]
    def quality_order(row):
        return -row["pass_rate"], row["mean_generation_seconds"], row["analysis_id"]
    cohorts = []
    for cohort_id, members in sorted(groups.items()):
        valid = sorted((row for row in members if row["valid_for_selection"]), key=quality_order)
        entry = {"cohort_id": cohort_id, "settings": members[0]["comparable_settings"],
                 "candidate_analysis_ids": sorted(row["analysis_id"] for row in members),
                 "pareto_frontier": pareto_frontier(members), "quality_maximum": None,
                 "practical_choice": None, "latency_budget_operating_points": [],
                 "paired_quality_screens": [], "selection_scope": "this validation cohort only"}
        if valid:
            best = valid[0]
            eligible = [best]
            entry["quality_maximum"] = {"analysis_id": best["analysis_id"], "candidate_id": best["candidate_id"]}
            for row in valid[1:]:
                paired = compare(best, row)
                clears = paired["accuracy_difference_ci"][0] >= -quality_tolerance
                entry["paired_quality_screens"].append({**paired,
                    "exploratory_margin_screen_passed": clears, "allowed_accuracy_loss": quality_tolerance})
                if clears:
                    eligible.append(row)
            practical = min(eligible, key=lambda row: (row["mean_generation_seconds"], -row["pass_rate"], row["analysis_id"]))
            entry["practical_choice"] = {"analysis_id": practical["analysis_id"],
                "candidate_id": practical["candidate_id"], "policy": "fastest passing exploratory paired quality-margin screen",
                "statistical_noninferiority_established": False}
            for budget in budgets:
                feasible = [row for row in valid if row["mean_generation_seconds"] <= budget]
                chosen = min(feasible, key=quality_order) if feasible else None
                entry["latency_budget_operating_points"].append({"mean_generation_seconds_budget": budget,
                    "analysis_id": chosen["analysis_id"] if chosen else None,
                    "candidate_id": chosen["candidate_id"] if chosen else None})
        cohorts.append(entry)
    ar_references = [row for row in summaries if row["valid_for_selection"] and row["candidate"]["decoder"] == "ar"]
    gates = []
    for row in summaries:
        if not row["valid_for_selection"] or row["candidate"]["decoder"] == "ar":
            continue
        for reference in ar_references:
            left, right = reference["comparable_settings"], row["comparable_settings"]
            if (left["taskset_sha256"] != right["taskset_sha256"] or left["seeds"] != right["seeds"]
                    or left["evaluation_protocol"] != right["evaluation_protocol"]):
                continue
            paired = compare(reference, row)
            gates.append({**paired, "reference_candidate_id": reference["candidate_id"],
                "candidate_id": row["candidate_id"], "allowed_accuracy_loss": quality_tolerance,
                "empirical_accuracy_target_met": paired["accuracy_difference"] >= -quality_tolerance,
                "exploratory_paired_ci_target_met": paired["accuracy_difference_ci"][0] >= -quality_tolerance,
                "candidate_faster_than_ar": row["mean_generation_seconds"] < reference["mean_generation_seconds"],
                "comparison_scope": "cross-model task/seed-matched AR target, not a decoder-only causal comparison"})
    report = {"schema_version": SCHEMA_VERSION, "generated_unix": time.time(),
        "selection_split": "validation", "holdout_used_for_selection": False,
        "primary_selection_metric": "micro_functional_pass_rate",
        "sensitivity_metric": "macro_language_pass_rate; equal weight for each manifest language, not used for selection",
        "candidate_id_filter": sorted(candidate_ids) if candidate_ids is not None else None,
        "inputs": inputs, "summaries": sorted(summaries, key=lambda row: row["analysis_id"]),
        "cohorts": cohorts, "ar_accuracy_gates": gates,
        "statistical_policy": {"bootstrap_repetitions": bootstrap_repetitions,
            "bootstrap_seed": bootstrap_seed, "quality_tolerance": quality_tolerance,
            "paired_comparisons_reported": len(comparisons), "multiplicity_adjusted": False,
            "post_selection_uncertainty": True, "statistical_noninferiority_established": False,
            "limitation": STATISTICAL_LIMITATION},
        "metric_policy": "Functional pass/fail is the blind grader result, independent of completion certification. Latency averages include all cases.",
        "global_best_claim": False}
    if output is not None:
        save_report(report, output, write_svg=write_svg)
    return report


def save_report(report, output, *, write_svg=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"tradeoff_analysis.json", report)
    columns = ("analysis_id", "candidate_id", "cohort_id", "valid_for_selection", "expected_cases",
        "observed_cases", "passed", "pass_rate", "macro_language_pass_rate", "mean_generation_seconds", "p95_generation_seconds",
        "mean_actual_forward_calls", "forced_case_rate", "incomplete_output_count", "incomplete_output_known_cases",
        "incomplete_output_rate", "verification_certified_count", "unresolved_mask_tokens",
        "infrastructure_error_count", "missing_or_invalid_cases", "completion_qualified_pass_rate")
    with (output/"tradeoff_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["summaries"])
    if write_svg:
        (output/"tradeoff.svg").write_text(_svg(report))


def _svg(report):
    """Standalone small multiples: different comparable cohorts never share a frontier."""
    width, panel_height = 1000, 320
    cohorts = report["cohorts"]
    height = 55 + panel_height*max(1, len(cohorts))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" font-size="12" fill="#17212b">',
        '<text x="30" y="24" font-size="18">Validation tradeoff by comparable cohort</text>',
        '<text x="30" y="44">All-case generation latency; empirical frontier. Exploratory selection, no global-best claim.</text>']
    summaries = {row["analysis_id"]: row for row in report["summaries"]}
    for number, cohort in enumerate(cohorts):
        top = 65 + number*panel_height
        rows = [summaries[key] for key in cohort["candidate_analysis_ids"] if summaries[key]["valid_for_selection"]]
        settings = cohort["settings"]
        label = (f'{settings["model"]} | adapter={digest(settings["adapter_sha256"])[:8]} | '
                 f'T={settings["temperature"]}, p={settings["top_p"]}, tokens={settings["tokens"]}, '
                 f'prefix={settings["prefix"]}, seeds={settings["seeds"]}')
        parts.append(f'<text x="30" y="{top+10}">{html.escape(label)}</text>')
        xmax = max((row["mean_generation_seconds"] for row in rows), default=1)*1.08 or 1
        def point(row):
            return 70+840*row["mean_generation_seconds"]/xmax, top+240-190*row["pass_rate"]
        parts.append(f'<path d="M70 {top+40} V{top+240} H930" fill="none" stroke="#777"/>')
        for tick in (0, .25, .5, .75, 1):
            y = top+240-190*tick
            parts.append(f'<text x="25" y="{y+4}">{tick*100:g}%</text>')
        for tick in (0, .25, .5, .75, 1):
            x = 70+840*tick
            parts.append(f'<text x="{x}" y="{top+260}" text-anchor="middle">{xmax*tick:.2f}</text>')
        parts.append(f'<text x="430" y="{top+280}">Mean generation seconds (lower is faster)</text>')
        frontier = [summaries[key] for key in cohort["pareto_frontier"]]
        coordinates = " ".join(f"{x:.2f},{y:.2f}" for x,y in map(point, frontier))
        parts.append(f'<polyline points="{coordinates}" fill="none" stroke="#999" stroke-dasharray="4 3"/>')
        for row in rows:
            x,y = point(row)
            color = {"ar": "#2563eb", "standard": "#d97706", "dard": "#059669"}.get(row["candidate"]["decoder"], "#555")
            title = f'{row["candidate_id"]}: {row["passed"]}/{row["expected_cases"]}, {row["mean_generation_seconds"]:.4f}s, NFE {row["mean_actual_forward_calls"]:.2f}'
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}"><title>{html.escape(title)}</title></circle>')
    parts.append('</g></svg>')
    return "\n".join(parts)+"\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261105)
    parser.add_argument("--quality-tolerance", type=float, default=.03)
    parser.add_argument("--latency-budget", action="append", type=float)
    parser.add_argument("--no-svg", action="store_true")
    parser.add_argument("--candidate-id", action="append")
    args = parser.parse_args(argv)
    report = analyze_evaluations(args.phase, output=args.output,
        bootstrap_repetitions=args.bootstrap_repetitions, bootstrap_seed=args.bootstrap_seed,
        quality_tolerance=args.quality_tolerance,
        latency_budgets=args.latency_budget or (1.,2.,5.,10.), write_svg=not args.no_svg,
        candidate_ids=args.candidate_id)
    print(json.dumps({"candidates": len(report["summaries"]), "cohorts": len(report["cohorts"]),
                      "output": str(args.output), "selection_split": "validation"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
