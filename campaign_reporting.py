"""Descriptive totals across fixed GPU shards; no independent-sample claims."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import hashlib
import math
import random
from pathlib import Path

DEFAULT_BACKENDS = ("ar", "standard", "dard")


def _publish(path, content):
    from experiment_analysis import _write_text
    _write_text(path, content)


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None


def _sum(values):
    values = list(values)
    return sum(values) if values and all(_number(value) is not None for value in values) else None


def _ratio(numerator, denominator):
    return round(numerator / denominator, 6) if numerator is not None and denominator is not None and denominator > 0 else None


def summarize_campaign(plan, shards, coverage, output):
    """Aggregate counts and summed work, preserving incomplete/unknown outcomes.

    shards contain each worker's raw rows and its independently controlled
    diagnostic report. Speed ratios use sums over the entire planned paired
    cohort; they never average per-shard ratios or select successful cases only.
    """
    backends = plan.get("backends", DEFAULT_BACKENDS)
    if (not isinstance(backends, (list, tuple)) or not backends
            or len(set(backends)) != len(backends) or any(value not in DEFAULT_BACKENDS for value in backends)):
        raise ValueError("Campaign backends must be a nonempty unique subset of ar,standard,dard")
    expected = Counter()
    expected_pairs = Counter()
    for worker in plan["workers"]:
        for dataset, language, _ in worker["task_keys"]:
            expected_pairs["all"] += plan["repeats"]
            expected_pairs[f"{dataset}/{language}"] += plan["repeats"]
            for decoder in backends:
                expected[f"{decoder}/all"] += plan["repeats"]
                expected[f"{decoder}/{dataset}/{language}"] += plan["repeats"]
    rows_by_group, pairs_by_group, phases = defaultdict(list), defaultdict(list), defaultdict(list)
    raw_count = latest_count = superseded = 0
    retry_times = []
    for shard in shards:
        raw = shard["rows"]
        def key(row):
            return (row.get("decoder"), row.get("dataset"), row.get("language"),
                    row.get("task_id", row.get("instance_id")), row.get("repeat"), row.get("seed"))
        latest = {key(row): index for index, row in enumerate(raw)}
        raw_count += len(raw)
        latest_count += len(latest)
        superseded += len(raw) - len(latest)
        retry_times.extend(row.get("total_wall_s") for index, row in enumerate(raw) if latest[key(row)] != index)
        diagnostics = shard.get("diagnostics")
        diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
        normalized = {key(row): row for row in diagnostics.get("cases", [])}
        for index in latest.values():
            row = dict(raw[index])
            if plan["kind"] == "repository":
                row["final_pass"] = normalized.get(key(row), {}).get("final_pass")
                row["case_completed"] = row.get("status") in ("submitted", "step_budget", "tool_budget", "deadline")
            for suffix in ("all", f"{row.get('dataset')}/{row.get('language')}"):
                rows_by_group[f"{row.get('decoder')}/{suffix}"].append(row)
        for pair in diagnostics.get("pairs", []):
            for suffix in ("all", f"{pair['dataset']}/{pair['language']}"):
                pairs_by_group[f"{pair['comparison']}/{suffix}"].append(pair)
        for phase in shard.get("phases", []):
            phases[phase.get("decoder")].append(phase)
    summaries = {}
    for group in sorted(expected.keys() | rows_by_group.keys()):
        rows = rows_by_group[group]
        planned = expected.get(group, 0)
        known = [row for row in rows if type(row.get("final_pass")) is bool and row.get("case_completed") is True]
        passed = sum(row["final_pass"] for row in known)
        first = [row for row in known if type(row.get("first_pass")) is bool]
        totals = {field: _sum(row.get(field) for row in rows) for field in
                  ("generation_wall_s", "agent_wall_s", "total_wall_s", "offline_grading_wall_s")}
        complete = len(rows) == planned and len(known) == planned and planned > 0 and coverage["coverage_complete"]
        summaries[group] = {
            "expected_cases": planned, "recorded_latest_cases": len(rows), "evaluated_cases": len(known),
            "missing_cases": max(0, planned - len(rows)), "unknown_or_incomplete_cases": len(rows) - len(known),
            "passed_cases": passed, "failed_cases": len(known) - passed,
            "first_passed_cases": sum(row["first_pass"] for row in first), "first_evaluated_cases": len(first),
            "complete_evaluated_cohort": complete,
            "observed_final_pass_rate": _ratio(passed, len(known)),
            "full_cohort_final_pass_rate": _ratio(passed, planned) if complete else None,
            "observed_first_pass_rate": _ratio(sum(row["first_pass"] for row in first), len(first)),
            "summed_latest_case_seconds": totals,
            "successful_cases_per_summed_task_hour": _ratio(passed * 3600, totals["total_wall_s"]) if complete else None,
            "statuses": dict(Counter(row.get("final_status", row.get("status", "unknown")) for row in rows)),
            "failure_kinds": dict(Counter(row.get("failure_kind") for row in rows if row.get("failure_kind"))),
        }
        if group.endswith("/all"):
            observed = phases[group.split("/", 1)[0]]
            wall = _sum(phase.get("phase_wall_s") for phase in observed)
            summaries[group]["phase_accounting"] = {
                "recorded_phases": len(observed), "all_phase_times_recorded": wall is not None,
                "summed_phase_wall_s": wall,
                "model_load_s": _sum(phase.get("model_load_s") for phase in observed),
                "warmup_s": _sum(phase.get("warmup_s") for phase in observed),
                "successful_cases_per_summed_phase_hour": _ratio(passed * 3600, wall) if complete else None,
            }
    comparisons = {}
    for baseline, candidate in (("ar", "dard"), ("standard", "dard"), ("ar", "standard")):
        if baseline not in backends or candidate not in backends:
            continue
        for suffix, planned in sorted(expected_pairs.items()):
            name = f"{candidate}_vs_{baseline}/{suffix}"
            pairs = pairs_by_group[name]
            identities = [(p["dataset"], p["language"], p["task_id"], p["repeat"], p["seed"]) for p in pairs]
            complete = len(pairs) == planned and len(set(identities)) == planned and coverage["coverage_complete"]
            known = bool(pairs) and all(pair.get("final_outcome") in
                        ("both_pass", "both_fail", "regression", "improvement") for pair in pairs)
            eligible = complete and known and all(p.get("eligible_for_controlled_comparison") is True for p in pairs)
            issues = {issue for pair in pairs for issue in pair.get("control_issues", []) + pair.get("timing_issues", [])}
            if not complete:
                issues.add("incomplete_or_duplicate_paired_cohort")
            if pairs and not known:
                issues.add("unknown_final_outcomes")
            outcomes = Counter(pair.get("final_outcome", "unknown") for pair in pairs)
            first_outcomes = Counter(pair.get("first_outcome", "unknown") for pair in pairs)
            first_known = bool(pairs) and all(pair.get("first_outcome") in
                ("both_pass", "both_fail", "regression", "improvement") for pair in pairs)
            comparisons[name] = {
                "expected_pairs": planned, "matched_pairs": len(pairs), "complete_cohort": complete,
                "eligible_for_controlled_comparison": eligible, "issues": sorted(issues),
                "final_outcomes": dict(outcomes),
                "first_outcomes": dict(first_outcomes),
                f"first_pass_rate_delta_{candidate}_minus_baseline":
                    (first_outcomes["improvement"] - first_outcomes["regression"]) / planned if eligible and first_known else None,
                f"final_pass_rate_delta_{candidate}_minus_baseline":
                    (outcomes["improvement"] - outcomes["regression"]) / planned if eligible and known else None,
                "speedup_ratio_of_summed_times": {field: _ratio(
                    _sum(pair["baseline"]["metrics"].get(field) for pair in pairs),
                    _sum(pair["candidate"]["metrics"].get(field) for pair in pairs)) if eligible else None
                    for field in ("first_generation_wall_s", "generation_wall_s", "agent_wall_s", "total_wall_s")},
                "pooled_confidence_interval": None,
            }
    status_path = Path(output) / "execution_status.json"
    invocation = json.loads(status_path.read_text()) if status_path.exists() else {}
    report = {
        "schema_version": 1, "kind": plan["kind"], "task_variants": plan["task_count"],
        "backends": list(backends),
        "expected_cases": plan["expected_cases"], "recorded_latest_cases": latest_count,
        "raw_case_rows": raw_count, "coverage_complete": coverage["coverage_complete"],
        "groups": summaries, "paired_comparisons": comparisons,
        "retry_accounting": {"superseded_result_rows": superseded,
                             "superseded_recorded_case_wall_s": _sum(retry_times) if superseded else 0},
        "latest_invocation_wall_s": invocation.get("invocation_wall_s"),
        "notes": [
            "Counts represent task/language/evaluator variants and repeated seeds, not independent problems.",
            "Speedups are ratios of summed paired work across all fixed GPU shards, never means of shard speedups or successful-only subsets.",
            "Groups retain incorrect answers and their time; infrastructure outcomes remain unknown rather than failed answers.",
            "Latest-case times exclude earlier infrastructure retries. Phase accounting includes recorded load, warmup, monitoring and retry costs; absent phase times remain unknown.",
            "Summed task/phase hours are accumulated worker time, not elapsed four-GPU campaign hours.",
            "The latest invocation can resume old results; no cumulative campaign throughput is inferred from its elapsed time.",
            "No pooled confidence intervals or architecture-only significance claim are provided.",
            "These results apply to the fixed recorded decoding budgets; they are not a quality/latency frontier sweep or a reproduction of the DARD paper.",
        ],
    }
    path = Path(output) / "campaign_summary.json"
    _publish(path, json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return report


def summarize_campaign_collection(campaigns, output, *, required_campaigns=("priority",)):
    """Index independent campaign reports without inventing shared controls.

    A priority AR/DARD run remains complete even when the optional standard
    campaign has no time left. Its original task identities, phases and paired
    eligibility stay in its own report; cross-run ratios are not synthesized.
    """
    required = set(required_campaigns)
    if not required.issubset(campaigns):
        raise ValueError("Every required campaign must be supplied")
    items = {}
    for name, directory in campaigns.items():
        directory = Path(directory).resolve()
        source = directory / "campaign_summary.json"
        report = json.loads(source.read_text(encoding="utf-8")) if source.exists() else None
        groups = report.get("groups", {}) if report else {}
        all_groups = [group for key, group in groups.items() if key.endswith("/all")]
        complete = bool(report and report.get("coverage_complete") and all_groups
                        and all(group.get("complete_evaluated_cohort") is True for group in all_groups))
        items[name] = {
            "required": name in required, "source": str(source),
            "available": report is not None, "complete_evaluated_campaign": complete,
            "report": report,
        }
    result = {
        "schema_version": 1, "campaigns": items,
        "required_campaigns_complete": all(items[name]["complete_evaluated_campaign"] for name in required),
        "all_available_campaigns_complete": all(item["complete_evaluated_campaign"] for item in items.values()),
        "cross_campaign_comparisons": [],
        "notes": [
            "Priority and optional campaign source reports retain their recorded backend selections, phases and controls.",
            "Optional standard work does not change completeness of the required AR/DARD campaign.",
            "No shared configuration or controlled cross-run comparison is inferred by combining these reports.",
        ],
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "combined_campaign_summary.json"
    _publish(path, json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return result


def write_priority_index(root):
    """Write a recoverable entry point for required and optional campaign work."""
    root = Path(root).resolve()
    result = summarize_campaign_collection({
        "priority": root / "functions-priority", "standard": root / "functions-standard",
    }, root)
    budget_path = root / "budget_decision.json"
    result["budget_decision_source"] = str(budget_path)
    result["budget_decision"] = json.loads(budget_path.read_text(encoding="utf-8")) if budget_path.exists() else None
    result["summary_groups"] = {}
    for name, item in result["campaigns"].items():
        report = item["report"] or {}
        result["summary_groups"][name] = {
            "backends": report.get("backends", []),
            "expected_cases": report.get("expected_cases"),
            "recorded_latest_cases": report.get("recorded_latest_cases"),
            "latest_invocation_wall_s": report.get("latest_invocation_wall_s"),
            "groups": {key: group for key, group in report.get("groups", {}).items() if key.endswith("/all")},
        }
    lines = ["# 우선 실험 결과", "",
             "AR·DARD 전체 평가를 우선 진행하며, standard diffusion은 남은 시간에 수행합니다.", "",
             f"필수 AR·DARD 평가 완료: {'예' if result['required_campaigns_complete'] else '아니오'}.", "",
             "| 실행 | 방식 | 계획 | 기록 | 채점 완료 | 정답 | 오답 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, item in result["campaigns"].items():
        report = item["report"] or {}
        for key, group in report.get("groups", {}).items():
            if not key.endswith("/all"):
                continue
            lines.append(f"| {name} | {key.removesuffix('/all')} | {group['expected_cases']} | "
                         f"{group['recorded_latest_cases']} | {group['evaluated_cases']} | "
                         f"{group['passed_cases']} | {group['failed_cases']} |")
    lines += ["", "미채점·실행 오류는 오답으로 합산하지 않습니다. 정답 수는 마지막 제출의 채점 결과입니다.", "",
              "| 실행 | 방식 | 생성 시간 합계(초) | 작업 시간 합계(초) | 로딩·준비 포함 phase 시간 합계(초) |",
              "| --- | --- | ---: | ---: | ---: |"]
    def seconds(value):
        return f"{value:.3f}" if _number(value) is not None else "미측정"
    for name, summary in result["summary_groups"].items():
        for key, group in summary["groups"].items():
            times = group["summed_latest_case_seconds"]
            phase = group.get("phase_accounting", {})
            lines.append(f"| {name} | {key.removesuffix('/all')} | {seconds(times['generation_wall_s'])} | "
                         f"{seconds(times['total_wall_s'])} | {seconds(phase.get('summed_phase_wall_s'))} |")
    lines += ["", "합산 시간은 GPU 작업자들의 누적 시간이며 실제 경과 시간과 다릅니다. "
              "서로 다른 실행의 standard·AR·DARD 측정값으로 통제된 3방식 비교를 만들지 않습니다.", ""]
    for name, item in result["campaigns"].items():
        label = "기록 있음" if item["available"] else "아직 결과 없음"
        elapsed = result["summary_groups"][name]["latest_invocation_wall_s"]
        lines.append(f"- [{name} 원본 결과]({item['source']}): {label}, 최근 실행 경과 {seconds(elapsed)}초.")
    lines += ["", f"[시간 예산 판단 원본]({budget_path})", "", "```json",
              json.dumps(result["budget_decision"], indent=2, ensure_ascii=False, sort_keys=True), "```", ""]
    for filename, content in (("priority_summary.json", json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"),
                              ("priority_summary.md", "\n".join(lines))):
        path = root / filename
        _publish(path, content)
    return result


def write_goal_report(campaign_output, *, bootstrap_draws=2000):
    """Report the single-submission quality/latency objective on one frozen run.

    The selected subset is fixed by campaign.json, never selected from outcomes.
    Missing cases/grades or mismatched controls cannot yield comparative numbers.
    Source-cluster resampling keeps translations of one problem together.
    """
    from experiment_analysis import analyze_run
    from function_benchmarks import OFFICIAL_SUITES
    from reporting import _percentile
    if not isinstance(bootstrap_draws, int) or isinstance(bootstrap_draws, bool) or bootstrap_draws < 1:
        raise ValueError("bootstrap_draws must be a positive integer")
    output = Path(campaign_output).resolve()
    plan = json.loads((output / "campaign.json").read_text(encoding="utf-8"))
    if plan.get("kind") != "functions":
        raise ValueError("The quality/latency goal report requires a function campaign")
    backends = tuple(plan.get("backends", DEFAULT_BACKENDS))
    issues = set()
    if len(backends) != 3 or set(backends) != set(DEFAULT_BACKENDS):
        issues.add("requires_all_three_backends")
    if plan.get("repeats") != 1:
        issues.add("requires_one_trial_per_task")
    expected, cases, pairs = set(), [], []
    configs = []
    duplicate_expected = False
    shard_reports = []
    for worker in plan["workers"]:
        run = Path(worker["output"])
        config_path = run / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        settings = config.get("settings", {})
        configs.append(settings)
        environments = settings.get("local_environments", {})
        if not isinstance(environments, dict):
            environments = {}
        for dataset, language, _ in worker["task_keys"]:
            required_environments = [language]
            if dataset in OFFICIAL_SUITES:
                required_environments.append("official:" + OFFICIAL_SUITES[dataset]["evaluator"])
            if any(not environments.get(name) for name in required_environments):
                issues.add("missing_required_local_environment_identity")
        if settings.get("repairs") != 0:
            issues.add("requires_zero_repairs")
        if settings.get("trial_seeds") is not None and len(settings["trial_seeds"]) != 1:
            issues.add("requires_one_trial_per_task")
        if run.exists():
            report = analyze_run(run)
            cases.extend(report["cases"])
            pairs.extend(report["pairs"])
            shard_reports.append(str(run / "diagnostics.json"))
        for task, seeds in zip(worker["task_keys"], worker["task_seeds"], strict=True):
            for repeat, seed in enumerate(seeds):
                for decoder in backends:
                    key = (decoder, *task, repeat, seed)
                    duplicate_expected |= key in expected
                    expected.add(key)
    if duplicate_expected or len(expected) != plan.get("expected_cases"):
        issues.add("invalid_or_duplicate_planned_cohort")
    # Shards may use different physical GPUs, but their generation protocol,
    # evaluator environment and source must agree before pooling their cases.
    controls = ("tokens", "repairs", "temperature", "top_p", "standard_steps", "dard_settings",
                "quant", "model", "ar_model", "revision", "ar_revision", "source_sha256")
    if configs and any(any(config.get(key) != configs[0].get(key) for key in controls) for config in configs[1:]):
        issues.add("different_shard_protocol_or_environment")
    # A small cohort can put only C++ on one shard and only Python/EvalPlus on
    # another. Compare identities for overlapping environment keys, not the
    # entire scope of installed runtimes recorded by each shard's preflight.
    environment_identities = {}
    for config in configs:
        environments = config.get("local_environments", {})
        if not isinstance(environments, dict):
            continue
        for name, identity_record in environments.items():
            if name in environment_identities and environment_identities[name] != identity_record:
                issues.add("different_shard_protocol_or_environment")
            environment_identities[name] = identity_record
    def identity(row):
        return (row["decoder"], row["dataset"], row["language"], row["task_id"], row["repeat"], row["seed"])
    counts = Counter(identity(row) for row in cases)
    if any(count != 1 for count in counts.values()) or set(counts) - expected:
        issues.add("duplicate_or_unplanned_recorded_cases")
    indexed = {identity(row): row for row in cases}
    if any(row["controls"]["case_completed"] is True and (row.get("attempts") != 1
           or (type(row["first_pass"]) is bool and type(row["final_pass"]) is bool
               and row["first_pass"] != row["final_pass"])) for row in cases):
        issues.add("inconsistent_single_submission_record")
    languages = sorted({key[2] for key in expected})
    if set(languages) != {"py", "js", "rs", "cpp"}:
        issues.add("requires_all_four_languages")
    groups = {}
    source_sets = {}
    def timing(rows, field):
        values = [row["metrics"].get(field) for row in rows]
        known = [value for value in values if _number(value) is not None]
        return {"recorded_cases": len(rows), "measured_cases": len(known),
                "sum_s": sum(known) if known else None,
                "mean_s": sum(known) / len(known) if known else None,
                "median_s": _percentile(known, .5), "p95_s": _percentile(known, .95),
                "all_recorded_cases_measured": bool(rows) and len(known) == len(rows)}
    for language in ["all", *languages]:
        for decoder in backends:
            planned = {key for key in expected if key[0] == decoder and (language == "all" or key[2] == language)}
            rows = [indexed[key] for key in sorted(planned) if key in indexed]
            known = [row for row in rows if row["controls"]["case_completed"] is True
                     and type(row["first_pass"]) is bool and type(row["final_pass"]) is bool]
            complete = bool(planned) and len(known) == len(planned) and not issues
            groups[f"{decoder}/{language}"] = {
                "expected_cases": len(planned), "recorded_cases": len(rows), "evaluated_cases": len(known),
                "first_pass_count": sum(row["first_pass"] for row in known),
                "complete_known_cohort": complete,
                "pass_at_1": sum(row["first_pass"] for row in known) / len(planned) if complete else None,
                "generation_latency": timing(rows, "first_generation_wall_s"),
                "end_to_end_latency": timing(rows, "total_wall_s"),
            }
        source_sets[language] = {row["source_id"] for row in cases if row.get("source_id")
                                 and (language == "all" or row["language"] == language)}
    comparisons = {}
    pair_index = defaultdict(list)
    for pair in pairs:
        pair_index[pair["comparison"]].append(pair)
    for baseline, candidate in (("ar", "dard"), ("standard", "dard"), ("ar", "standard")):
        for language in ["all", *languages]:
            name = f"{candidate}_vs_{baseline}/{language}"
            matched = [pair for pair in pair_index[f"{candidate}_vs_{baseline}"]
                       if language == "all" or pair["language"] == language]
            wanted = {key[1:] for key in expected if key[0] == baseline and (language == "all" or key[2] == language)}
            wanted_candidate = {key[1:] for key in expected if key[0] == candidate and (language == "all" or key[2] == language)}
            found = [(p["dataset"], p["language"], p["task_id"], p["repeat"], p["seed"]) for p in matched]
            cohort_complete = bool(wanted) and wanted == wanted_candidate and set(found) == wanted and len(found) == len(wanted)
            local_issues = set(issues)
            if not cohort_complete:
                local_issues.add("incomplete_paired_cohort")
            for pair in matched:
                local_issues.update(pair["control_issues"] + pair["timing_issues"])
                if any(type(pair[side]["first_pass"]) is not bool for side in ("baseline", "candidate")):
                    local_issues.add("unknown_first_outcomes")
                if (not pair["baseline"].get("source_id")
                        or pair["baseline"].get("source_id") != pair["candidate"].get("source_id")):
                    local_issues.add("missing_or_different_source_cluster")
                if not pair.get("eligible_for_controlled_comparison"):
                    local_issues.add("ineligible_case_controls")
            eligible = cohort_complete and not local_issues
            delta = ci = generation_speed = task_speed = None
            clusters = defaultdict(list)
            for pair in matched:
                clusters[pair["baseline"].get("source_id")].append(pair)
            if eligible:
                delta = sum(int(pair["candidate"]["first_pass"]) - int(pair["baseline"]["first_pass"])
                            for pair in matched) / len(matched)
                if len(clusters) >= 2:
                    totals = [(len(group), sum(int(p["candidate"]["first_pass"]) - int(p["baseline"]["first_pass"])
                                              for p in group)) for group in clusters.values()]
                    rng = random.Random(int(hashlib.sha256(name.encode()).hexdigest()[:16], 16))
                    samples = []
                    for _ in range(bootstrap_draws):
                        draw = [rng.choice(totals) for _ in totals]
                        samples.append(sum(value for _, value in draw) / sum(count for count, _ in draw))
                    ci = [_percentile(samples, .025), _percentile(samples, .975)]
                generation_speed = _ratio(_sum(p["baseline"]["metrics"].get("first_generation_wall_s") for p in matched),
                                          _sum(p["candidate"]["metrics"].get("first_generation_wall_s") for p in matched))
                task_speed = _ratio(_sum(p["baseline"]["metrics"].get("total_wall_s") for p in matched),
                                    _sum(p["candidate"]["metrics"].get("total_wall_s") for p in matched))
            comparisons[name] = {"expected_pairs": len(wanted), "matched_pairs": len(matched),
                "complete_known_controlled_cohort": eligible, "issues": sorted(local_issues),
                "unique_source_clusters": len(clusters), "pass_at_1_delta_candidate_minus_baseline": delta,
                "pass_at_1_delta_source_cluster_bootstrap_95ci": ci,
                "generation_speedup_baseline_over_candidate": generation_speed,
                "end_to_end_speedup_baseline_over_candidate": task_speed,
                "quality_similarity_margin": None, "quality_similarity_established": None,
                "speed_and_quality_goal_established": None}
    coverage_complete = bool(expected) and set(counts) == expected and all(count == 1 for count in counts.values())
    complete_evaluated = (coverage_complete and set(backends) == set(DEFAULT_BACKENDS)
                          and all(groups.get(f"{decoder}/all", {}).get("complete_known_cohort") is True
                                  for decoder in DEFAULT_BACKENDS))
    report = {"schema_version": 1, "campaign": str(output), "selection_manifest": str(output / "campaign.json"),
        "shard_diagnostics": shard_reports, "selection_policy": plan.get("selection_manifest"),
        "backends": list(backends), "languages": languages,
        "coverage_complete": coverage_complete, "complete_evaluated_campaign": complete_evaluated,
        "protocol": "one_complete_function_submission_per_task_no_repair",
        "protocol_issues": sorted(issues), "selected_task_variants": plan.get("task_count"),
        "selected_unique_source_problems": len(source_sets["all"]),
        "common_source_problems_across_languages": len(set.intersection(*(source_sets[lang] for lang in languages))) if languages else 0,
        "groups": groups, "comparisons": comparisons, "bootstrap_draws": bootstrap_draws,
        "quality_similarity_margin": None, "overall_goal_established": None,
        "notes": [
            "pass_at_1 is observed single-submission full-function correctness under this runner, not official completion-only pass@1.",
            "Each comparison uses all planned paired outcomes, including incorrect answers; missing/infrastructure outcomes never become failures or zeros.",
            "The paired correctness CI resamples original source problems with all translations/evaluator variants together. It is descriptive and not a population guarantee.",
            "No acceptable quality-loss margin was supplied, so quality similarity and the combined speed/quality goal are not declared established.",
            "Generation latency and end-to-end task time are separate. End-to-end includes final grading and any recorded public checking; model loading/warmup and earlier infrastructure retries are excluded.",
            "Timing sums accumulate work across GPUs, not campaign elapsed time. Ratios greater than one favor the named candidate.",
            "Results apply only to the preselected budgeted cohort, frozen checkpoints and decoding budgets; they do not establish full-benchmark or architecture-only superiority.",
            "Source overlap across languages is reported explicitly; translated tasks are correlated, and pooled accuracy weights task variants rather than unique source problems.",
            "Python and translated languages can use different graders even for the same source IDs; language pass-rate differences alone do not establish language ability differences.",
        ]}
    lines = ["# 코드 생성 정확도·속도 결과", "", "한 문제에 한 번 생성한 결과를 비교합니다. 수정 재시도와 반복 생성은 포함하지 않습니다.", "",
             "| 언어 | 방식 | 채점/계획 | 정답률 | 생성 평균/중앙값/p95(초) | 작업 평균(초) |",
             "| --- | --- | ---: | ---: | --- | ---: |"]
    def value(number, *, percent=False):
        return "미확정" if number is None else (f"{number * 100:.2f}%" if percent else f"{number:.3f}")
    for language in languages:
        for decoder in backends:
            group = groups[f"{decoder}/{language}"]
            latency = group["generation_latency"]
            lines.append(f"| {language} | {decoder} | {group['evaluated_cases']}/{group['expected_cases']} | "
                         f"{value(group['pass_at_1'], percent=True)} | "
                         + "/".join(value(latency[key]) for key in ("mean_s", "median_s", "p95_s"))
                         + f" | {value(group['end_to_end_latency']['mean_s'])} |")
    lines += ["", "| 비교/언어 | 정답률 차이(후보−기준, %p) | source cluster 95% CI(%p) | 생성 배속 | 작업 배속 |",
              "| --- | ---: | --- | ---: | ---: |"]
    for name, comparison in comparisons.items():
        ci = comparison["pass_at_1_delta_source_cluster_bootstrap_95ci"]
        delta = comparison['pass_at_1_delta_candidate_minus_baseline']
        lines.append(f"| {name} | {value(delta * 100 if delta is not None else None)} | "
                     + (" ~ ".join(value(bound * 100) for bound in ci) if ci else "미확정")
                     + f" | {value(comparison['generation_speedup_baseline_over_candidate'])} | "
                     + f"{value(comparison['end_to_end_speedup_baseline_over_candidate'])} |")
    lines += ["", "배속이 1보다 크면 후보 방식이 빠릅니다. 허용 가능한 정확도 저하 폭이 정해지지 않아 "
              "‘비슷한 품질’ 달성을 선언하지 않습니다. 미완료·미채점·통제 조건 불일치 비교는 미확정으로 표시합니다.", "",
              f"선택된 문제 변형 {plan.get('task_count')}건, 원문 문제 {len(source_sets['all'])}개. "
              "결론은 선택된 문제와 고정 설정에 한정됩니다.", "",
              "같은 원문 문제라도 Python과 번역 언어의 채점기는 다를 수 있어 언어별 정답률만으로 언어 능력을 비교할 수 없습니다.", "",
              "생성 시간과 채점 포함 작업 시간을 분리했습니다. GPU 간 누적 시간은 실제 경과 시간이 아닙니다.", "",
              f"[상세 수치·제약]({output / 'goal_report.json'}) · [선택 계획]({output / 'campaign.json'})", ""]
    for filename, content in (("goal_report.json", json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"),
                              ("goal_report.md", "\n".join(lines))):
        path = output / filename
        _publish(path, content)
    return report
