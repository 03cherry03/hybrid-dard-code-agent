"""Read-only analysis of paired function/repository experiments; no inference.

Raw evidence stays in the run logs. This report contains numeric summaries and
references, never generated source, gold patches or hidden test contents. Runs
are analyzed independently, including when --run is repeated.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import html
import json
import math
import os
from pathlib import Path
import re

from experiment_journal import configuration_fingerprints, read_journal


IDENTITY_FIELDS = ("model_name", "resolved_revision", "tokenizer_revision",
                   "tokenizer_vocab_sha256", "model_config_sha256", "chat_template_sha256")
HARDWARE_FIELDS = ("gpu_uuid", "cuda_build", "versions", "compute_dtype", "quant")
COUNTERS = ("model_forward_passes", "forced_tokens", "promoted_tokens", "revoked_tokens",
            "demoted_tokens", "u_reverified_tokens", "forced_forward_passes",
            "attention_fallback_steps", "output_slots", "generated_tokens",
            "visible_output_tokens", "unfilled_mask_tokens")
TIMINGS = ("generation_wall_s", "model_wall_s", "first_generation_wall_s", "agent_wall_s",
           "total_wall_s", "test_wall_s", "offline_grading_wall_s", "tool_wall_s",
           "setup_wall_s", "cleanup_wall_s")


def _number(value):
    return float(value) if (isinstance(value, (int, float)) and not isinstance(value, bool)
                            and math.isfinite(value)) else None


def _sum(values):
    values = list(values)
    numbers = [_number(value) for value in values]
    return sum(numbers) if numbers and all(v is not None for v in numbers) else None


def _ratio(left, right):
    left, right = _number(left), _number(right)
    return round(left / right, 6) if left is not None and right is not None and right > 0 else None


def _delta(left, right):
    left, right = _number(left), _number(right)
    return round(right - left, 6) if left is not None and right is not None else None


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_text(path, content):
    """Publish a completed analysis artifact without a partially replaced file."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _jsonl(run, name):
    path = run / name
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Expected an object")
        except ValueError as exc:
            raise ValueError(f"{name}:{number}: invalid JSON object; analysis did not skip it") from exc
        rows.append({**row, "_evidence": f"{name}:{number}"})
    return rows


def _key(row):
    return (row.get("dataset"), row.get("language"), row.get("task_id", row.get("instance_id")),
            row.get("repeat"), row.get("seed"))


def _full_key(row):
    return (row.get("decoder"), *_key(row))


def _repo_key(row):
    return (row.get("dataset"), row.get("instance_id"), row.get("decoder"), row.get("repeat"))


def _aggregate_generation(row, events):
    generations = [event["generation"] for event in events if isinstance(event.get("generation"), dict)]
    result = {name: _number(row.get(name)) for name in (*COUNTERS, *TIMINGS)}
    for name in COUNTERS:
        if result[name] is None:
            result[name] = _sum(g.get(name) for g in generations)
    if result["model_wall_s"] is None:
        result["model_wall_s"] = _sum(g.get("model_wall_s") for g in generations)
    if result["first_generation_wall_s"] is None and generations:
        result["first_generation_wall_s"] = _number(generations[0].get("wall_s"))
    observed_peaks = [_number(g.get("gpu_peak_gib")) for g in generations]
    observed_peaks = [value for value in observed_peaks if value is not None]
    row_peak = _number(row.get("max_gpu_peak_gib", row.get("gpu_peak_gib")))
    result["max_gpu_peak_gib"] = row_peak if row_peak is not None else max(observed_peaks, default=None)
    result["gpu_peak_calls_observed"] = len(observed_peaks) if generations else None
    # Some runner summaries predate nullable metrics and fill absent generation
    # measurements with zero. Raw interrupted-call records take precedence over
    # those defaults. Parent-measured wall times remain valid even after a kill.
    interrupted = any(g.get("measurement_complete") is False or
        (g.get("diagnostics") or {}).get("complete") is False for g in generations)
    result["partial_generation_measurements"] = {}
    if interrupted:
        for name in (*COUNTERS, "model_wall_s"):
            values = [_number(g.get(name)) for g in generations]
            observed = [value for value in values if value is not None]
            result[name] = sum(observed) if len(observed) == len(generations) else None
            result["partial_generation_measurements"][name] = {
                "observed_sum": sum(observed) if observed else None,
                "calls_observed": len(observed), "calls_recorded": len(generations),
                "all_calls_have_measurement": len(observed) == len(generations),
                "contains_interrupted_call": True,
            }
        result["max_gpu_peak_gib"] = max(observed_peaks) if len(observed_peaks) == len(generations) else None
        result["partial_generation_measurements"]["max_gpu_peak_gib"] = {
            "observed_max": max(observed_peaks, default=None),
            "calls_observed": len(observed_peaks), "calls_recorded": len(generations),
            "all_calls_have_measurement": len(observed_peaks) == len(generations),
            "contains_interrupted_call": True,
        }
    result["generation_calls"] = len(generations) if events else None
    result["attempts_or_steps"] = row.get("attempts", row.get("steps"))
    result["invalid_actions"] = (row.get("invalid_action_count") if "invalid_action_count" in row else
        sum(bool(e.get("action_parse_error")) or str(e.get("observation", {}).get("error", "")).startswith("Invalid action:")
            for e in events) if events else None)
    result["tool_errors"] = (row.get("tool_error_count") if "tool_error_count" in row else
        sum(e.get("action", {}).get("action") not in (None, "finish") and
            e.get("observation", {}).get("ok") is False for e in events) if events else None)
    result["context_trim_events"] = (row["context_trim_events"] if "context_trim_events" in row else
        sum(bool(e.get("context", {}).get("dropped_history_messages")) for e in events)
        if events and all("context" in e for e in events) else None)
    result["tool_output_truncations"] = (sum(bool(e["tool_output"].get("source_truncated") or
        e["tool_output"].get("context_truncated")) for e in events if "tool_output" in e)
        if any("tool_output" in e for e in events) else None)
    result["patch_changed_files"] = row.get("patch_changed_files")
    result["patch_bytes"] = row.get("patch_bytes")
    result["budget_exhausted_calls"] = (sum(g["budget_exhausted"] is True for g in generations)
        if generations and all("budget_exhausted" in g for g in generations) else None)
    result["stop_reasons"] = dict(Counter(g["stop_reason"] for g in generations if g.get("stop_reason")))
    stages, shapes, decoder_phases = defaultdict(list), defaultdict(list), defaultdict(list)
    modes = set()
    measured = []
    for generation in generations:
        diagnostics = generation.get("diagnostics") or {}
        mode = generation.get("diagnostics_mode", diagnostics.get("mode"))
        if mode:
            modes.add(mode)
        if "measurement_complete" in generation:
            measured.append(generation["measurement_complete"] is True)
        elif "complete" in diagnostics:
            measured.append(diagnostics["complete"] is True)
        for name, value in diagnostics.get("stage_wall_s", {}).items():
            if _number(value) is not None:
                stages[name].append(value)
        for name, value in diagnostics.get("forward_shapes", {}).items():
            if _number(value) is not None:
                shapes[name].append(value)
        decoder = diagnostics.get("decoder") or {}
        for name, value in decoder.get("component_wall_s", decoder.get("phase_wall_s", {})).items():
            if _number(value) is not None:
                decoder_phases[name].append(value)
    result["stage_wall_s"] = {k: {"sum": sum(v), "calls_observed": len(v)} for k, v in stages.items()}
    result["forward_shapes"] = {k: {"max": max(v), "sum": sum(v), "calls_observed": len(v)} for k, v in shapes.items()}
    result["decoder_phase_wall_s"] = {k: {"sum": sum(v), "calls_observed": len(v)} for k, v in decoder_phases.items()}
    if row.get("diagnostics_mode"):
        modes.add(row["diagnostics_mode"])
    result["diagnostics_modes"] = sorted(modes)
    result["instrumentation_affects_timing"] = bool(row.get("instrumentation_affects_timing") or
        any(g.get("instrumentation_affects_timing") or g.get("diagnostics", {}).get("instrumentation_affects_timing")
            for g in generations) or any(mode != "summary" for mode in modes))
    result["measurement_complete"] = row.get("measurement_complete", all(measured) if measured else None)
    result["forced_assignments_per_output_slot"] = _ratio(result["forced_tokens"], result["output_slots"])
    result["generation_seconds_per_forward"] = _ratio(result["generation_wall_s"], result["model_forward_passes"])
    return result


def _normalize(row, events, config, kind, grade=None):
    phase = next((p for p in config.get("phases", []) if p.get("phase_id") == row.get("phase_id")), {})
    metadata = row.get("model_metadata", phase.get("backend", {}))
    if not metadata:
        metadata = config.get("backend_identities", {}).get("ar" if row.get("decoder") == "ar" else "diffusion", {})
    evidence = [row["_evidence"], *[e["_evidence"] for e in events if e.get("_evidence")]]
    if grade:
        evidence.append(grade["_evidence"])
    metrics = _aggregate_generation(row, events)
    return {
        "decoder": row.get("decoder"), "dataset": row.get("dataset"), "language": row.get("language"),
        "task_id": row.get("task_id", row.get("instance_id")), "source_id": row.get("source_id"),
        "repeat": row.get("repeat"), "seed": row.get("seed"),
        "attempts": row.get("attempts"),
        "task_sha256": row.get("task_sha256"),
        "prompt_sha256": row.get("messages_sha256", row.get("initial_messages_sha256")),
        "first_pass": row.get("first_pass") if kind == "function" else None,
        "final_pass": row.get("final_pass") if kind == "function" else (grade or {}).get("resolved"),
        "status": row.get("final_status", row.get("status")),
        "grade_status": (grade or {}).get("status"),
        "failure_kind": row.get("failure_kind"),
        "first_error_category": row.get("first_error_category"), "last_error_category": row.get("last_error_category"),
        "evidence": list(dict.fromkeys(evidence)), "metrics": metrics,
        "controls": {
            "model": {k: metadata.get(k) for k in IDENTITY_FIELDS},
            "hardware": {k: metadata.get(k) for k in HARDWARE_FIELDS},
            "model_identity_verified": row.get("model_identity_verified", metadata.get("identity_verified")),
            "warmup_done": row.get("warmup_done", config.get("warmup", 0) > 0),
            "gpu_checked": row.get("resource_check_available") if kind == "function" else
                all(row.get(k, {}).get("verified") for k in ("gpu_before", "gpu_after")),
            "gpu_contended": row.get("resource_contended") if kind == "function" else
                any(row.get(k, {}).get("other_pids") for k in ("gpu_before", "gpu_after")),
            "environment_id": row.get("environment_id"), "case_completed": row.get("case_completed", row.get("status") in
                ("submitted", "step_budget", "tool_budget", "deadline")),
            "deadline_violation": row.get("deadline_violation", False),
            "first_generation_budget": events[0].get("generation", {}).get("generation_budget_tokens") if events else None,
        },
    }


def _controls(left, right, config, kind):
    issues = []
    settings = config.get("settings", config)
    if not config or not settings.get("task_manifest_sha256", settings.get("task_set_sha256")):
        issues.append("missing_experiment_manifest")
    if not settings.get("environment", settings.get("runtime")) or not settings.get("source_sha256"):
        issues.append("missing_environment_or_source_fingerprint")
    budget_fields = ("tokens", "repairs", "temperature", "top_p") if kind == "function" else ("agent_config", "temperature", "top_p")
    if any(k not in settings for k in budget_fields):
        issues.append("missing_shared_budget_configuration")
    if kind == "repository" and any(k not in settings.get("agent_config", {}) for k in
            ("max_context_chars", "max_input_tokens", "max_output_tokens", "max_steps", "max_tool_calls",
             "task_timeout_s", "tool_timeout_s", "max_tool_output_chars")):
        issues.append("missing_shared_context_or_tool_budget_configuration")
    for field in ("task_sha256", "prompt_sha256"):
        if not left[field] or left[field] != right[field]:
            issues.append("missing_or_different_" + field)
    if any(value is None for value in _key(left)):
        issues.append("missing_pair_identity")
    a, b = left["controls"], right["controls"]
    if not all(c["model_identity_verified"] and all(c["model"].values()) for c in (a, b)):
        issues.append("unresolved_model_or_tokenizer_identity")
    if left["decoder"] == "standard" and a["model"] != b["model"]:
        issues.append("different_diffusion_checkpoint_or_tokenizer")
    if any(c["hardware"].get(k) is None for c in (a, b) for k in HARDWARE_FIELDS):
        issues.append("missing_hardware_precision_or_library_identity")
    elif a["hardware"] != b["hardware"]:
        issues.append("different_hardware_precision_or_libraries")
    if kind == "repository" and (not a["environment_id"] or a["environment_id"] != b["environment_id"]):
        issues.append("missing_or_different_repository_environment")
    if a["first_generation_budget"] is not None and b["first_generation_budget"] is not None and a["first_generation_budget"] != b["first_generation_budget"]:
        issues.append("different_generation_token_budget")
    if any((case["metrics"]["attention_fallback_steps"] or 0) > 0 for case in (left, right)):
        issues.append("diagnostic_attention_fallback")
    timing = []
    if not all(c["gpu_checked"] for c in (a, b)):
        timing.append("gpu_process_check_unavailable")
    if any(c["gpu_contended"] for c in (a, b)):
        timing.append("other_compute_process_on_selected_gpu")
    if not all(c["warmup_done"] for c in (a, b)):
        timing.append("missing_warmup")
    if any(c["deadline_violation"] for c in (a, b)):
        timing.append("deadline_censored_timing")
    if not all(c["case_completed"] for c in (a, b)):
        timing.append("incomplete_case")
    if any(case["metrics"]["instrumentation_affects_timing"] for case in (left, right)):
        timing.append("instrumentation_affects_timing")
    if any(case["metrics"]["measurement_complete"] is False for case in (left, right)):
        timing.append("incomplete_measurement")
    return issues, timing


def _outcome(left, right):
    if type(left) is not bool or type(right) is not bool:
        return "unknown"
    if left != right:
        return "regression" if left else "improvement"
    return "both_pass" if left else "both_fail"


def _compare(left, right, config, kind):
    issues, timing_issues = _controls(left, right, config, kind)
    if type(left["final_pass"]) is not bool or type(right["final_pass"]) is not bool:
        timing_issues.append("unknown_final_outcomes")
    a, b = left["metrics"], right["metrics"]
    eligible = not issues and not timing_issues
    result = {
        "comparison": right["decoder"] + "_vs_" + left["decoder"],
        "dataset": right["dataset"], "language": right["language"], "task_id": right["task_id"],
        "repeat": right["repeat"], "seed": right["seed"],
        "controls_match": not issues, "control_issues": issues, "timing_issues": timing_issues,
        "eligible_for_controlled_comparison": eligible,
        "first_outcome": _outcome(left["first_pass"], right["first_pass"]) if not issues else "uncontrolled",
        "final_outcome": _outcome(left["final_pass"], right["final_pass"]) if not issues else "uncontrolled",
        "observed_generation_speedup_baseline_over_dard": _ratio(a["generation_wall_s"], b["generation_wall_s"]) if eligible else None,
        "observed_end_to_end_speedup_baseline_over_dard": _ratio(a["total_wall_s"], b["total_wall_s"]) if eligible else None,
        "observed_agent_speedup_baseline_over_dard": _ratio(a["agent_wall_s"], b["agent_wall_s"]) if eligible else None,
        "forward_difference_dard_minus_baseline": _delta(a["model_forward_passes"], b["model_forward_passes"]) if not issues else None,
        "observed_generation_difference_s": _delta(a["generation_wall_s"], b["generation_wall_s"]) if eligible else None,
        "baseline": left, "candidate": right,
        "evidence": list(dict.fromkeys(left["evidence"] + right["evidence"])),
        "quality_interpretation_issues": (["instrumentation_can_change_outcomes_under_time_limits"]
            if a["instrumentation_affects_timing"] or b["instrumentation_affects_timing"] else []),
        "speed_or_quality_improvement_proven": False,
    }
    for field in ("observed_generation_speedup_baseline_over_dard", "observed_end_to_end_speedup_baseline_over_dard",
                  "observed_agent_speedup_baseline_over_dard", "forward_difference_dard_minus_baseline"):
        result[field.replace("dard", "candidate")] = result[field]
        if right["decoder"] != "dard":
            result[field] = None
    return result


def _findings(pair):
    if pair["candidate"]["decoder"] != "dard":
        return []
    a, b = pair["baseline"]["metrics"], pair["candidate"]["metrics"]
    findings = []
    def add(code, evidence, hypothesis, experiment):
        findings.append({"code": code, "task_id": pair["task_id"], "repeat": pair["repeat"],
            "comparison": pair["comparison"], "observed": evidence, "hypothesis": hypothesis,
            "next_development_experiment": experiment, "evidence": pair["evidence"],
            "causal_claim": False})
    if pair["control_issues"] or pair["timing_issues"]:
        add("comparison_controls_incomplete", pair["control_issues"] + pair["timing_issues"],
            "The current records cannot isolate a clean speed comparison.",
            "Resolve the listed control/measurement issue and repeat the entire preselected development cohort.")
    if pair["forward_difference_dard_minus_baseline"] is not None and pair["forward_difference_dard_minus_baseline"] < 0 and (pair["observed_generation_difference_s"] or 0) > 0:
        add("fewer_forwards_but_slower", {"forwards": [a["model_forward_passes"], b["model_forward_passes"]],
            "generation_s": [a["generation_wall_s"], b["generation_wall_s"]]},
            "Reduced forward count did not reduce measured time; packed input/attention size or decoder bookkeeping may dominate.",
            "Use a separate profile run on fixed development inputs to locate the cost; then vary only block_size with all other settings fixed, and remeasure summary mode.")
    if pair["final_outcome"] == "regression" or pair["first_outcome"] == "regression":
        add("quality_regression", {"first": pair["first_outcome"], "final": pair["final_outcome"],
            "forced_tokens": b["forced_tokens"], "revoked_tokens": b["revoked_tokens"],
            "budget_exhausted_calls": b["budget_exhausted_calls"]},
            "This matched problem lost correctness; forcing, confidence decisions or truncation are candidate explanations, not established causes.",
            "Inspect the linked generation and failure records on development tasks. Predeclare one threshold or budget change, and rerun all selected development tasks including previous successes.")
    if (b["forced_tokens"] or 0) > 0:
        add("forced_completion_observed", {"forced_tokens": b["forced_tokens"], "output_slots": b["output_slots"]},
            "The decoder reached forced completion; unverified assignments may affect quality. This remains part of the evaluated algorithm.",
            "On development tasks, vary only steps_per_block to check whether forcing and correctness change together; include the added time cost.")
    if (b["revoked_tokens"] or 0) > 0:
        add("revocation_work_observed", {"revoked_tokens": b["revoked_tokens"], "demoted_tokens": b["demoted_tokens"]},
            "Revisions consumed work; they may repair mistakes or indicate unstable candidate decisions.",
            "Inspect step traces in a separate development diagnostic run; change only one confidence threshold in the next run and compare quality as well as time.")
    if (b["budget_exhausted_calls"] or 0) > 0 or (b["unfilled_mask_tokens"] or 0) > 0:
        add("output_budget_or_mask_incomplete", {"budget_exhausted_calls": b["budget_exhausted_calls"],
            "unfilled_mask_tokens": b["unfilled_mask_tokens"]},
            "Output limits or incomplete denoising may leave unusable code/actions; a budget stop alone does not prove truncation.",
            "Inspect stop reasons on development tasks, then increase only the shared output-token budget for all compared backends.")
    if (b["invalid_actions"] or 0) > 0:
        add("invalid_repository_actions", {"invalid_actions": b["invalid_actions"], "steps": b["attempts_or_steps"]},
            "Malformed tool requests consumed agent steps and may offset generation speed.",
            "Inspect the linked public action errors; test one shared action-format prompt change on a fixed development cohort across all backends.")
    if (b["tool_errors"] or 0) > 0:
        add("repository_tool_errors", {"tool_errors": b["tool_errors"], "tool_s": b["tool_wall_s"]},
            "Failed tool calls may consume time without advancing the patch; command correctness, environment support or time limits could be involved.",
            "Inspect public development tool observations and categorize the failure first; test one shared tool-protocol or tool-timeout change while holding the task budget fixed.")
    if (b["context_trim_events"] or 0) > 0:
        add("repository_context_trimmed", {"context_trim_events": b["context_trim_events"]},
            "The bounded conversation dropped earlier observations; repeated work or missed constraints may follow, but the trace is needed to establish that connection.",
            "Inspect dropped-history counts and subsequent public actions on development tasks; vary only the shared context budget within model limits, then measure quality, time and memory.")
    if (b["tool_output_truncations"] or 0) > 0:
        add("repository_tool_output_truncated", {"tool_output_truncations": b["tool_output_truncations"]},
            "Some command output was shortened before the agent received it, which may omit useful diagnostics or merely remove irrelevant text.",
            "Inspect public development tool-output metadata; change only the shared output cap at the observed truncation stage, keeping conversation limits fixed.")
    if ((pair["observed_generation_speedup_baseline_over_dard"] or 0) > 1
            and pair["observed_end_to_end_speedup_baseline_over_dard"] is not None
            and pair["observed_end_to_end_speedup_baseline_over_dard"] < 1):
        add("generation_gain_lost_end_to_end", {"generation_speedup": pair["observed_generation_speedup_baseline_over_dard"],
            "end_to_end_speedup": pair["observed_end_to_end_speedup_baseline_over_dard"],
            "tool_s": [a["tool_wall_s"], b["tool_wall_s"]], "test_s": [a["test_wall_s"], b["test_wall_s"]],
            "steps": [a["attempts_or_steps"], b["attempts_or_steps"]]},
            "Non-generation work or additional agent steps offset the generation improvement.",
            "Locate the extra tool/test/repair cost in traces; select one shared scaffold change and rerun the complete development cohort.")
    return findings


def _error_summary(run, kind, config, rows, events):
    """Record stage/type and provenance only; never expose exception bodies/tests."""
    records = []
    def stage_name(value):
        return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", value) else "unknown"
    def error_type(value):
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)(?=[:(]|$)", str(value or ""))
        return match[1] if match and (match[1].endswith(("Error", "Exception", "Interrupt"))
                                    or match[1] in ("SystemExit", "GeneratorExit")) else "unknown"
    def add(stage, error, evidence, task=None):
        records.append({"stage": stage_name(stage), "error_type": error_type(error),
                        "evidence": evidence, "task_id": task})
    if kind == "function":
        for error in _jsonl(run, "errors.jsonl"):
            diagnostics = error.get("generation_diagnostics", {}).get("diagnostics", {})
            add(error.get("stage", diagnostics.get("phase", "generation_or_evaluation")),
                error.get("error_type", error.get("exception")), error["_evidence"], error.get("task_id"))
        for index, phase in enumerate(config.get("phases", [])):
            if phase.get("error"):
                add(phase.get("error_stage", "backend_phase"), phase["error"], f"config.json#phases[{index}]")
    elif kind == "repository":
        for row in rows:
            if row.get("error"):
                message = str(row["error"])
                stage = "model_load_or_warmup" if message.startswith("Model load/warmup failed:") else row.get("status", "unknown")
                add(stage, message.split(": ", 1)[-1] if stage == "model_load_or_warmup" else message,
                    row["_evidence"], row.get("instance_id"))
        for trace in events.values():
            for event in trace:
                if event.get("generation_error"):
                    add(event.get("generation", {}).get("diagnostics", {}).get("phase", "generation"),
                        event["generation_error"], event.get("_evidence"), None)
    breadcrumbs = {}
    for filename in ("worker_status.json", "current_case.json"):
        saved = _read_json(run / filename)
        if saved:
            breadcrumbs[filename] = {key: saved.get(key) for key in (
                "pid", "updated_unix", "phase_id", "decoder", "dataset", "language", "task_id",
                "repeat", "seed", "attempt", "exit_code", "case_completed") if key in saved}
            breadcrumbs[filename].update(stage=stage_name(saved.get("stage")), state=stage_name(saved.get("state")),
                                          error_type=error_type(saved.get("error_type")),
                                          evidence=filename, process_liveness_not_checked=True)
    return {"recorded_error_events": len(records), "stage_counts": dict(Counter(r["stage"] for r in records)),
            "type_counts": dict(Counter(r["error_type"] for r in records)), "records": records[:100],
            "last_durable_operation": breadcrumbs,
            "records_omitted": max(0, len(records) - 100),
            "counting_note": "Recorded events may refer to the same failure; counts are not distinct failure counts. Raw exception/test text is omitted. A running breadcrumb records the last persisted operation, not proof of a live process or the cause of a hard kill."}


def analyze_run(run: Path) -> dict:
    run = Path(run).resolve()
    if not run.is_dir():
        raise ValueError(f"Run directory does not exist: {run}")
    function = any((run / name).exists() for name in ("config.json", "case_results.jsonl", "runs.jsonl", "errors.jsonl"))
    repository = (run / "run.json").exists() or (run / "results.jsonl").exists()
    if function and repository:
        raise ValueError("Ambiguous run directory contains function and repository manifests")
    kind = "function" if function else "repository" if repository else "unknown"
    config_name = "config.json" if function else "run.json"
    config = _read_json(run / config_name)
    raw = _jsonl(run, "case_results.jsonl" if function else "results.jsonl") if kind != "unknown" else []
    rows = list({_full_key(r): r for r in raw}.values())
    if kind == "repository" and len(rows) != len(raw):
        raise ValueError("Duplicate repository results are ambiguous; refusing to choose a successful retry")
    events = defaultdict(list)
    all_attempts = []
    grades = {}
    if kind == "function":
        all_attempts = _jsonl(run, "runs.jsonl")
        for event in all_attempts:
            key = _full_key(event)
            # Function retries restart at attempt 1. Keep the most recent attempt sequence.
            if event.get("attempt") == 1:
                events[key] = []
            events[key].append(event)
    elif kind == "repository":
        for bundle in _jsonl(run, "traces.jsonl"):
            for row in rows:
                if tuple(bundle.get("key", [])) == _repo_key(row):
                    events[_full_key(row)] = [{**event, "_evidence": bundle["_evidence"] + f"#step={event.get('step', index + 1)}"}
                        for index, event in enumerate(bundle.get("trace", []))]
        grades = {_repo_key(r): r for r in _jsonl(run, "grades.jsonl")}
    cases = [_normalize(row, events[_full_key(row)], config, kind, grades.get(_repo_key(row))) for row in rows]
    indexed = {_full_key(case): case for case in cases}
    pairs, unmatched = [], []
    expected = config.get("expected_groups", {})
    if kind == "repository":
        for task in config.get("tasks", []):
            for decoder in config.get("decoders", []):
                key = f"{decoder}/{task.get('dataset')}/{task.get('language')}"
                expected[key] = expected.get(key, 0) + config.get("repeats", 0)
    datasets = {(r["dataset"], r["language"]) for r in cases}
    datasets.update(tuple(k.split("/", 2)[1:]) for k in expected if len(k.split("/", 2)) == 3)
    groups = {}
    for dataset, language in sorted(datasets):
        for baseline, candidate in (("standard", "dard"), ("ar", "dard"), ("ar", "standard")):
            relevant = [r for r in cases if r["dataset"] == dataset and r["language"] == language and r["decoder"] in (baseline, candidate)]
            keys = {_key(r) for r in relevant}
            current = []
            for key in sorted(keys, key=str):
                left, right = indexed.get((baseline, *key)), indexed.get((candidate, *key))
                if not left or not right:
                    unmatched.append({"comparison": candidate + "_vs_" + baseline, "key": list(key),
                                      "missing_decoder": baseline if not left else candidate,
                                      "evidence": (left or right)["evidence"]})
                    continue
                current.append(_compare(left, right, config, kind))
            pairs.extend(current)
            planned = [expected.get(f"{d}/{dataset}/{language}") for d in (baseline, candidate)]
            complete = bool(current) and all(count is not None and count == len(current) for count in planned)
            eligible = complete and all(p["eligible_for_controlled_comparison"] for p in current)
            known = bool(current) and all(p["final_outcome"] in ("regression", "improvement", "both_pass", "both_fail") for p in current)
            outcomes = dict(Counter(p["final_outcome"] for p in current))
            group_key = f"{candidate}_vs_{baseline}/{dataset}/{language}"
            groups[group_key] = {"matched_pairs": len(current), "expected_baseline": planned[0], "expected_candidate": planned[1], "expected_dard": planned[1] if candidate == "dard" else None,
                "complete_cohort": complete, "eligible_for_controlled_comparison": eligible,
                "final_outcomes": outcomes, "first_outcomes": dict(Counter(p["first_outcome"] for p in current)),
                f"final_pass_rate_delta_{candidate}_minus_baseline":
                    (outcomes.get("improvement", 0) - outcomes.get("regression", 0)) / len(current) if known and complete else None,
                "generation_speedup_baseline_over_candidate": _ratio(_sum(p["baseline"]["metrics"]["generation_wall_s"] for p in current),
                    _sum(p["candidate"]["metrics"]["generation_wall_s"] for p in current)) if eligible else None,
                "end_to_end_speedup_baseline_over_candidate": _ratio(_sum(p["baseline"]["metrics"]["total_wall_s"] for p in current),
                    _sum(p["candidate"]["metrics"]["total_wall_s"] for p in current)) if eligible else None,
                "issues": sorted({issue for p in current for issue in p["control_issues"] + p["timing_issues"]}
                    | ({"incomplete_paired_evaluation"} if not complete else set())),
                "findings": [finding for p in current for finding in _findings(p)],
                "speed_or_quality_improvement_proven": False}
            groups[group_key]["final_pass_rate_delta_candidate_minus_baseline"] = groups[group_key][f"final_pass_rate_delta_{candidate}_minus_baseline"]
            if candidate == "dard":
                groups[group_key]["generation_speedup_baseline_over_dard"] = groups[group_key]["generation_speedup_baseline_over_candidate"]
                groups[group_key]["end_to_end_speedup_baseline_over_dard"] = groups[group_key]["end_to_end_speedup_baseline_over_candidate"]
    journal = read_journal(run)
    context = journal[0]["details"] if journal else {}
    latest_evidence = {row["_evidence"] for row in rows}
    superseded = [row for row in raw if row["_evidence"] not in latest_evidence]
    active_attempts = {event["_evidence"] for trace in events.values() for event in trace if "_evidence" in event}
    return {"schema_version": 1, "run": str(run), "kind": kind,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "no_evaluation_data" if not cases else "analysis_available",
        "configuration_fingerprints": configuration_fingerprints(run),
        "experiment_context": context, "journal_entries": len(journal),
        "error_summary": _error_summary(run, kind, config, raw, events),
        "retry_history": {"superseded_result_rows": len(superseded),
            "superseded_attempt_records": sum(e["_evidence"] not in active_attempts for e in all_attempts),
            "records": [{"evidence": r["_evidence"], "task_id": r.get("task_id"), "decoder": r.get("decoder"),
                "repeat": r.get("repeat"), "seed": r.get("seed"), "failure_kind": r.get("failure_kind"),
                "case_completed": r.get("case_completed")} for r in superseded[:100]],
            "records_omitted": max(0, len(superseded) - 100),
            "timing_policy": "Superseded retry records are preserved as provenance; their times are not pooled into latest-case comparisons."},
        "cases": cases, "pairs": pairs, "unmatched": unmatched, "groups": groups,
        "notes": [
            "Ratios greater than 1 favor the named candidate. All outcomes including failures stay in the selected cohort.",
            "Missing measurements/grades are null or unknown, never inferred as zero or test failure.",
            "Only equal task/repeat/seed and matching recorded controls form interpretable pairs; no cross-run measurements are pooled.",
            "Generation count is not FLOPs; packed attention length, context, output length, and per-forward work can differ.",
            "Forced completion and revocations remain included as algorithm behavior; diagnostic fallback and detailed timing runs cannot establish clean speed comparisons.",
            "Function total time includes final grading; repository total time includes setup/cleanup and excludes official grading. Model load/warmup are separate.",
            "Hypotheses are candidates for investigation, not causal conclusions. Use separate profile runs; confirm performance in summary mode.",
            "Profile/steps instrumentation can also change quality outcomes under fixed time limits. Those outcomes are descriptive only until reproduced in summary mode.",
            "Change one factor on a fixed development cohort. Freeze the chosen settings before held-out evaluation; never tune against held-out failures.",
            "Translations, enhanced test variants, and repeated seeds are correlated observations; this report provides no independent-sample significance claim.",
            "A dev/test journal label is declared intent; users must select disjoint development and held-out tasks.",
        ]}


def _safe(value):
    return html.escape(str(value)).replace("|", "&#124;").replace("\n", " ")


def render_markdown(report):
    lines = ["# Experiment diagnostics", "", f"Run: `{_safe(report['run'])}`", "",
             f"Status: **{report['status']}**. Results below are descriptive; hypotheses do not establish causes.", ""]
    if report["experiment_context"]:
        lines.append("Declared experiment intent: " + _safe(json.dumps(report["experiment_context"], ensure_ascii=False)))
        lines.append("")
    if not report["cases"]:
        lines += ["No evaluation case records exist. Accuracy, speed and failure causes are unknown.", ""]
    errors = report["error_summary"]
    if errors["recorded_error_events"]:
        lines += ["## Recorded errors", "", f"Recorded error events: {errors['recorded_error_events']} (may refer to the same failure). "
            + "Stages: " + _safe(json.dumps(errors["stage_counts"])) + ". Exception bodies are omitted.", ""]
        lines += [f"- {_safe(r['stage'])}: {_safe(r['error_type'])}; evidence {_safe(r['evidence'])}" for r in errors["records"]]
        lines.append("")
    if report["retry_history"]["superseded_result_rows"] or report["retry_history"]["superseded_attempt_records"]:
        history = report["retry_history"]
        lines += [f"Earlier retry records: {history['superseded_result_rows']} result rows and {history['superseded_attempt_records']} attempt records; their timings are not pooled. See retry_history in diagnostics.json.", ""]
    for name, group in report["groups"].items():
        lines += [f"## {_safe(name)}", "", f"Matched pairs: {group['matched_pairs']}; complete cohort: {group['complete_cohort']}; controlled timing eligible: {group['eligible_for_controlled_comparison']}.", "",
            "Final outcomes: " + _safe(json.dumps(group["final_outcomes"])) + ".", "",
            f"Generation speed ratio: {group['generation_speedup_baseline_over_candidate']}; end-to-end ratio: {group['end_to_end_speedup_baseline_over_candidate']}.", ""]
        if group["issues"]:
            lines += ["Control/coverage issues: " + ", ".join(_safe(x) for x in group["issues"]), ""]
        for finding in group["findings"][:50]:
            lines += [f"- **{_safe(finding['code'])}** — task {_safe(finding['task_id'])}, repeat {finding['repeat']}. "
                + "Observed: " + _safe(json.dumps(finding["observed"], ensure_ascii=False))
                + " Hypothesis: " + _safe(finding["hypothesis"])
                + " Next development check: " + _safe(finding["next_development_experiment"])
                + " Evidence: " + ", ".join(_safe(e) for e in finding["evidence"])]
        if len(group["findings"]) > 50:
            lines.append(f"- {len(group['findings']) - 50} additional findings are retained in diagnostics.json.")
        lines.append("")
    if report["unmatched"]:
        lines += [f"Unmatched task/seed rows: {len(report['unmatched'])}. See diagnostics.json; they are not silently dropped.", ""]
    lines += ["## Interpretation", "", *["- " + note for note in report["notes"]], ""]
    return "\n".join(lines)


def write_report(run: Path, report=None) -> dict:
    run = Path(run)
    report = analyze_run(run) if report is None else report
    for name, content in (("diagnostics.json", json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"),
                          ("diagnostics.md", render_markdown(report))):
        _write_text(run / name, content)
    columns = ("comparison", "dataset", "language", "task_id", "repeat", "seed", "first_outcome", "final_outcome",
        "eligible_for_controlled_comparison", "observed_generation_speedup_baseline_over_dard",
        "observed_end_to_end_speedup_baseline_over_dard", "forward_difference_dard_minus_baseline",
        "observed_generation_speedup_baseline_over_candidate", "observed_end_to_end_speedup_baseline_over_candidate",
        "forward_difference_candidate_minus_baseline")
    csv_path = run / "diagnostics_cases.csv"
    csv_temporary = csv_path.with_name(csv_path.name + ".tmp")
    with csv_temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for pair in report["pairs"]:
            # Avoid spreadsheet formula interpretation of externally supplied task IDs.
            writer.writerow({key: ("'" + pair[key] if isinstance(pair[key], str) and pair[key].startswith(("=", "+", "-", "@")) else pair[key]) for key in columns})
        stream.flush()
        os.fsync(stream.fileno())
    csv_temporary.replace(csv_path)
    directory = os.open(run, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    try:
        for run in args.run:
            report = write_report(run)
            print(json.dumps({"run": str(run), "status": report["status"],
                "cases": len(report["cases"]), "pairs": len(report["pairs"]), "report": str(run / "diagnostics.md")}))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
