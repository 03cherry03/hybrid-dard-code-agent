"""Descriptive, paired reports for the common code-generation/repair protocol."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import random
import statistics
from pathlib import Path


def case_key(row):
    return (row["decoder"], row["dataset"], row["language"], row["task_id"],
            row.get("repeat", 0), row.get("seed", 42))


def latest_cases(path: Path) -> list[dict]:
    """Append-only retries replace an unfinished case in the materialized report."""
    rows = {}
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid case log line {index + 1}; repair the interrupted JSONL before resuming") from exc
            rows[case_key(row)] = row
    return list(rows.values())


def _percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (pos - low), 6)


def _timings(rows, field):
    values = [float(r[field]) for r in rows if r.get(field) is not None]
    return {"mean": round(statistics.mean(values), 6) if values else None,
            "p50": _percentile(values, .5), "p95": _percentile(values, .95),
            "sum": round(sum(values), 6)}


def _ratio(numerator, denominator):
    return round(numerator / denominator, 6) if denominator > 0 else None


def _paired_comparison(baseline, candidate, *, expected, expected_ar, manifest_available,
                       tolerance, draws, random_seed):
    def key(row):
        return (row["task_id"], row.get("repeat", 0), row.get("seed", 42))
    left = {key(r): r for r in baseline}
    right = {key(r): r for r in candidate}
    common = sorted(left.keys() & right.keys())
    pairs = [(left[k], right[k]) for k in common
             if left[k].get("case_completed", True) and right[k].get("case_completed", True)
             and left[k]["first_pass"] is not None and right[k]["first_pass"] is not None
             and left[k]["final_pass"] is not None and right[k]["final_pass"] is not None]
    n = len(pairs)
    first_delta = sum(int(b["first_pass"]) - int(a["first_pass"]) for a, b in pairs) / n if n else None
    final_delta = sum(int(b["final_pass"]) - int(a["final_pass"]) for a, b in pairs) / n if n else None
    jointly_passed = [(a, b) for a, b in pairs if a["final_pass"] and b["final_pass"]]
    prompt_mismatches = sum(a.get("messages_sha256") != b.get("messages_sha256") for a, b in pairs)
    missing_prompts = sum(not a.get("messages_sha256") or not b.get("messages_sha256") for a, b in pairs)
    task_mismatches = sum(not a.get("task_sha256") or a.get("task_sha256") != b.get("task_sha256") for a, b in pairs)
    unresolved_models = sum(not a.get("model_identity_verified", False) or
                            not b.get("model_identity_verified", False) for a, b in pairs)
    fallback_cases = sum(bool(b.get("attention_fallback_steps", 0)) for _, b in pairs)
    contended = sum(bool(a.get("resource_contended") or b.get("resource_contended")) for a, b in pairs)
    unchecked = sum(not a.get("resource_check_available", False) or
                    not b.get("resource_check_available", False) for a, b in pairs)
    # Cluster by original task: repeated seeds/timings are not independent problems.
    clusters = defaultdict(list)
    for a, b in pairs:
        clusters[a.get("source_id") or a["task_id"]].append((a, b))
    rng = random.Random(random_seed)
    accuracy_samples, speed_samples = [], []
    cluster_values = list(clusters.values())
    if len(cluster_values) >= 2:
        totals = [(len(group),
                   sum(int(b["first_pass"]) - int(a["first_pass"]) for a, b in group),
                   sum(a.get("first_generation_wall_s", a["generation_wall_s"]) for a, _ in group),
                   sum(b.get("first_generation_wall_s", b["generation_wall_s"]) for _, b in group))
                  for group in cluster_values]
        for _ in range(draws):
            sampled = [rng.choice(totals) for _ in totals]
            count = sum(x[0] for x in sampled)
            accuracy_samples.append(sum(x[1] for x in sampled) / count)
            denom = sum(x[3] for x in sampled)
            if denom > 0:
                speed_samples.append(sum(x[2] for x in sampled) / denom)
    issues = []
    if expected is None or expected_ar is None or n != expected or n != expected_ar:
        issues.append("incomplete_paired_evaluation")
    if not manifest_available:
        issues.append("missing_experiment_manifest")
    if prompt_mismatches:
        issues.append("different_shared_prompt_content")
    if missing_prompts:
        issues.append("missing_prompt_fingerprint")
    if task_mismatches:
        issues.append("missing_or_different_task_fingerprint")
    if unresolved_models:
        issues.append("unresolved_model_or_tokenizer_identity")
    if fallback_cases:
        issues.append("dard_diagnostic_attention_fallback")
    if contended:
        issues.append("other_compute_process_on_selected_gpu")
    if unchecked:
        issues.append("gpu_process_check_unavailable")
    if any(r.get("instrumentation_affects_timing") or r.get("diagnostics_mode", "summary") != "summary"
           for r in baseline + candidate):
        issues.append("diagnostic_instrumentation_affects_timing")
    if any(r.get("measurement_complete") is False for r in baseline + candidate):
        issues.append("incomplete_generation_measurements")
    if any(not a.get("warmup_done", False) or not b.get("warmup_done", False) for a, b in pairs):
        issues.append("no_warmup")
    return {
        "paired_evaluated_cases": n, "expected_pairs": expected, "expected_ar_cases": expected_ar,
        "unique_problem_clusters": len(clusters),
        "first_pass_rate_delta_candidate_minus_ar": round(first_delta, 6) if n else None,
        "final_pass_rate_delta_candidate_minus_ar": round(final_delta, 6) if n else None,
        "observed_first_pass_within_tolerance": first_delta >= -tolerance if n else None,
        "accuracy_tolerance": tolerance,
        "initial_generation_speedup_ar_over_candidate": _ratio(
            sum(a.get("first_generation_wall_s", a["generation_wall_s"]) for a, _ in pairs),
            sum(b.get("first_generation_wall_s", b["generation_wall_s"]) for _, b in pairs)),
        "total_task_speedup_ar_over_candidate": _ratio(
            sum(a.get("total_wall_s", a["generation_wall_s"] + a["test_wall_s"]) for a, _ in pairs),
            sum(b.get("total_wall_s", b["generation_wall_s"] + b["test_wall_s"]) for _, b in pairs)),
        "jointly_successful_cases": len(jointly_passed),
        "jointly_successful_total_time_speedup": _ratio(
            sum(a.get("total_wall_s", a["generation_wall_s"] + a["test_wall_s"]) for a, _ in jointly_passed),
            sum(b.get("total_wall_s", b["generation_wall_s"] + b["test_wall_s"]) for _, b in jointly_passed)),
        "first_pass_delta_cluster_bootstrap_95ci": [_percentile(accuracy_samples, .025), _percentile(accuracy_samples, .975)],
        "generation_speedup_cluster_bootstrap_95ci": [_percentile(speed_samples, .025), _percentile(speed_samples, .975)],
        "bootstrap_draws": draws, "comparison_issues": issues,
        "eligible_for_controlled_comparison": n > 0 and not issues,
        "interpretation": "Descriptive comparison of these checkpoints and this runner; not proof of architecture-only effects or statistical equivalence.",
    }


def summarize(results: list[dict], metadata: dict, output: Path):
    groups = defaultdict(list)
    for row in results:
        groups[(row["decoder"], row["dataset"], row["language"])].append(row)
    expected_groups = metadata.get("expected_groups", {})
    for key in expected_groups:
        groups.setdefault(tuple(key.split("/", 2)), [])
    summaries = {}
    for (dec, dataset, lang), rows in sorted(groups.items()):
        evaluated = [r for r in rows if r["first_pass"] is not None and r["final_pass"] is not None
                     and r.get("case_completed", True)]
        initial = sum(r["first_pass"] is True for r in evaluated)
        final = sum(r["final_pass"] is True for r in evaluated)
        repaired = sum(r["first_pass"] is False and r["final_pass"] is True for r in evaluated)
        key = f"{dec}/{dataset}/{lang}"
        total_time = sum(r.get("total_wall_s", r["generation_wall_s"] + r["test_wall_s"]) for r in rows)
        summaries[key] = {
            "cases": len(rows), "expected_cases": expected_groups.get(key, len(rows)),
            "evaluated": len(evaluated), "not_evaluated": len(rows) - len(evaluated),
            "completed": sum(r.get("case_completed", True) for r in rows),
            "diagnostics_modes": dict(Counter(r.get("diagnostics_mode", "unknown") for r in rows)),
            "instrumented_cases": sum(bool(r.get("instrumentation_affects_timing")) for r in rows),
            "incomplete_measurement_cases": sum(r.get("measurement_complete") is False for r in rows),
            "first_pass_count": initial if evaluated else None,
            "final_pass_count": final if evaluated else None,
            "first_pass_rate": round(initial / len(evaluated), 6) if evaluated else None,
            "final_pass_rate": round(final / len(evaluated), 6) if evaluated else None,
            "repair_success_given_initial_failure": _ratio(repaired, len(evaluated) - initial),
            "time_generation_s": round(sum(r["generation_wall_s"] for r in rows), 6),
            "time_test_s": round(sum(r["test_wall_s"] for r in rows), 6),
            "time_total_s": round(total_time, 6),
            "successful_tasks_per_hour": _ratio(final * 3600, total_time) if evaluated else None,
            "generation_latency_s": _timings(rows, "generation_wall_s"),
            "first_generation_latency_s": _timings(rows, "first_generation_wall_s"),
            "total_task_latency_s": _timings(rows, "total_wall_s"),
            "agent_latency_s": _timings(rows, "agent_wall_s"),
            "offline_grading_latency_s": _timings(rows, "offline_grading_wall_s"),
            "forward_passes": sum(r.get("model_forward_passes", 0) for r in rows),
            **{name: sum(r.get(name, 0) for r in rows) for name in
               ("forced_tokens", "promoted_tokens", "revoked_tokens", "attention_fallback_steps")},
            **{name: sum(r.get(name, 0) for r in rows) for name in
               ("demoted_tokens", "u_reverified_tokens", "forced_forward_passes")},
            "max_gpu_peak_gib": max((r.get("max_gpu_peak_gib", 0) for r in rows), default=0),
            "errors": dict(Counter(r["final_status"] for r in rows)),
            "sealed_error_categories": dict(Counter(r.get("last_error_category", "unknown") for r in rows)),
            "feedback_error_categories": dict(Counter(r.get("feedback_last_error_category", "unknown") for r in rows)),
        }
    comparisons = {}
    for dec, dataset, lang in sorted(groups):
        if dec == "ar":
            continue
        for baseline in (("ar", "standard") if dec == "dard" else ("ar",)):
            if (baseline, dataset, lang) not in groups:
                continue
            key = f"{dec}_vs_{baseline}/{dataset}/{lang}"
            seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
            identities = metadata.get("backend_identities", {})
            comparison = _paired_comparison(
                groups[(baseline, dataset, lang)], groups[(dec, dataset, lang)],
                expected=expected_groups.get(f"{dec}/{dataset}/{lang}"),
                expected_ar=expected_groups.get(f"{baseline}/{dataset}/{lang}"),
                manifest_available=bool(metadata.get("settings", {}).get("task_manifest_sha256")
                                        and identities.get("diffusion")
                                        and (baseline != "ar" or identities.get("ar"))),
                tolerance=metadata.get("accuracy_tolerance", 0.0),
                draws=metadata.get("bootstrap_draws", 1000), random_seed=seed)
            comparisons[key] = {name.replace("_ar", "_" + baseline): value for name, value in comparison.items()}
    families = defaultdict(list)
    for (dec, dataset, lang), rows in groups.items():
        family = rows[0].get("family", dataset) if rows else metadata.get("suite_families", {}).get(dataset, dataset)
        rate = summaries[f"{dec}/{dataset}/{lang}"]["first_pass_rate"]
        if rate is not None:
            evaluator = rows[0].get("final_evaluator", "legacy_assert") if rows else "legacy_assert"
            families[(dec, family, evaluator)].append((lang, rate))
    macro = {f"{dec}/{family}/{evaluator}": {
                "language_macro_first_pass_rate": round(statistics.mean(v for _, v in values), 6),
                "language_groups": len(values)} for (dec, family, evaluator), values in families.items()}
    report = {"metadata": metadata, "groups": summaries, "paired_comparisons": comparisons,
              "family_macro": macro,
              "notes": ["Translations of the same original problem are correlated; language and family results are reported separately.",
                        "In sealed_final_v1, repairs use only public feedback; final grades are computed after submission and never returned to the agent. The full-function protocol is not official completion-only pass@1.",
                        "Base and augmented test suites share original problems; evaluator variants are kept separate in family macro averages.",
                        "Agent time excludes sealed grading. Total task time adds final grading; offline first-attempt diagnostic grading is recorded separately, not counted as an agent action.",
                        "Timing/solved-tasks-per-hour use latest case attempts, excluding model loading, warmup, GPU monitoring and earlier infrastructure retries; phase times and append-only logs retain those costs.",
                        "Steps/profile diagnostics may change synchronization and logging costs. Their timing is diagnostic only and excluded from controlled speed comparisons; use summary in a separate run.",
                        "CI resamples original problems with all repeats together; single-problem groups have no CI."]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if results:
        fields = list(dict.fromkeys(key for row in results for key in row))
        with (output / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
    return report
