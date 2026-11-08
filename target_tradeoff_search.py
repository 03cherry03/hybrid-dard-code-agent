"""Check an explicit accuracy/latency target in already measured validation rows.

This module never runs inference or changes measurements. ``rows`` uses the
``summaries`` schema emitted by tradeoff_analysis.py. Raw results.json rows lack
task/seed/protocol identity and are deliberately not guessed into this schema.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path


DEFAULT_AR_ID = "ar-7f348bc3385bc1e4"
MATCHED_KEYS = ("model", "revision", "adapter_sha256", "tokens", "prefix",
                "temperature", "top_p")
EVALUATION_KEYS = ("taskset_sha256", "seeds", "evaluation_protocol")
PROTOCOL_KEYS = ("candidate_warmup", "counterbalanced_order", "extraction_version",
                 "generation_timeout", "grading_timeout")


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _integer(value):
    return type(value) is int and value >= 0


def _validate(row):
    if not isinstance(row, dict):
        return ["row_not_object"]
    errors = []
    cid, candidate = row.get("candidate_id"), row.get("candidate")
    if not isinstance(cid, str) or not cid:
        errors.append("missing_candidate_id")
    if not isinstance(candidate, dict):
        candidate = {}
        errors.append("missing_candidate")
    if candidate.get("id") != cid:
        errors.append("candidate_id_mismatch")
    if candidate.get("decoder") not in ("ar", "standard", "dard"):
        errors.append("unsupported_decoder")
    if row.get("valid_for_selection") is not True:
        errors.append("not_valid_for_selection")
    if row.get("split", "validation") != "validation":
        errors.append("not_validation")
    for key in ("missing_or_invalid_cases", "infrastructure_error_count"):
        if not _integer(row.get(key)) or row[key] != 0:
            errors.append(key + "_not_zero")
    expected, observed, passed = (row.get(k) for k in
                                 ("expected_cases", "observed_cases", "passed"))
    if not _integer(expected) or expected == 0:
        errors.append("invalid_expected_cases")
    if not _integer(observed) or observed == 0 or observed != expected:
        errors.append("incomplete_case_coverage")
    accuracy, latency = row.get("pass_rate"), row.get("mean_generation_seconds")
    if not _number(accuracy) or not 0 <= accuracy <= 1:
        errors.append("invalid_pass_rate")
    if not _number(latency) or latency <= 0:
        errors.append("invalid_mean_generation_seconds")
    if not _integer(passed) or not _integer(observed) or passed > observed:
        errors.append("invalid_passed_count")
    elif observed and _number(accuracy) and not math.isclose(
            accuracy, passed / observed, rel_tol=0, abs_tol=1e-12):
        errors.append("pass_rate_count_mismatch")
    settings = row.get("comparable_settings")
    if not isinstance(settings, dict):
        return errors + ["missing_comparable_settings"]
    for key in MATCHED_KEYS + EVALUATION_KEYS:
        if key not in settings:
            errors.append("missing_setting:" + key)
    for key in ("model", "revision", "taskset_sha256", "prefix"):
        if not isinstance(settings.get(key), str) or not settings[key]:
            errors.append("invalid_setting:" + key)
    for key in ("model", "revision", "tokens", "prefix"):
        if settings.get(key) != candidate.get(key):
            errors.append("candidate_setting_mismatch:" + key)
    sampling = candidate.get("dard", {})
    if not isinstance(sampling, dict):
        sampling = {}
    for key in ("temperature", "top_p"):
        if not _number(settings.get(key)) or settings.get(key) != sampling.get(key):
            errors.append("candidate_setting_mismatch:" + key)
    if _number(settings.get("temperature")) and settings["temperature"] < 0:
        errors.append("invalid_setting:temperature")
    if _number(settings.get("top_p")) and not 0 < settings["top_p"] <= 1:
        errors.append("invalid_setting:top_p")
    if not _integer(settings.get("tokens")) or settings["tokens"] == 0:
        errors.append("invalid_setting:tokens")
    adapter = candidate.get("adapter")
    adapter_hashes = adapter.get("sha256") if isinstance(adapter, dict) else None
    if adapter is not None and (not isinstance(adapter_hashes, dict) or not adapter_hashes):
        errors.append("missing_adapter_identity")
    if settings.get("adapter_sha256") != adapter_hashes:
        errors.append("candidate_setting_mismatch:adapter_sha256")
    seeds = settings.get("seeds")
    if (not isinstance(seeds, list) or not seeds or not all(_integer(x) for x in seeds)
            or len(set(seeds)) != len(seeds)):
        errors.append("invalid_seeds")
    elif row.get("seeds") is not None:
        outer_seeds = row["seeds"]
        if (not isinstance(outer_seeds, list) or not all(_integer(x) for x in outer_seeds)
                or sorted(outer_seeds) != sorted(seeds)):
            errors.append("seed_metadata_mismatch")
    protocol = settings.get("evaluation_protocol")
    if not isinstance(protocol, dict) or any(k not in protocol for k in PROTOCOL_KEYS):
        errors.append("incomplete_evaluation_protocol")
    else:
        if (any(type(protocol[k]) is not bool for k in ("candidate_warmup", "counterbalanced_order"))
                or not isinstance(protocol["extraction_version"], str) or not protocol["extraction_version"]
                or any(not _number(protocol[k]) or protocol[k] <= 0 for k in
                       ("generation_timeout", "grading_timeout"))):
            errors.append("invalid_evaluation_protocol")
    return errors


def _differences(left, right, keys):
    a, b = left["comparable_settings"], right["comparable_settings"]
    return [key for key in keys if (sorted(a[key]) if key == "seeds" else a[key])
            != (sorted(b[key]) if key == "seeds" else b[key])]


def _point(row):
    return {key: row[key] for key in ("candidate_id", "candidate", "pass_rate",
            "mean_generation_seconds", "expected_cases", "observed_cases", "passed",
            "comparable_settings")}


def evaluate_target(rows, ar_id=DEFAULT_AR_ID, max_accuracy_gap=0.05,
                    comparison_mode="matched"):
    """Assess strict AR>DARD>standard accuracy and standard<DARD<AR latency.

    The AR-to-DARD accuracy gap is inclusive (5pp means 0.05, not 5).
    ``matched`` holds checkpoint, adapter, length, prefix and sampling fixed
    between standard and DARD. ``operating_point`` explicitly relaxes those
    settings, but NEVER tasksets, seeds, evaluation protocols or case coverage.
    AR uses its own model/context, while its evaluation identity must match.
    Feasibility is descriptive, without a significance/noninferiority claim.
    ``best_gap_candidate`` is only a diagnostic closest comparable triad; it is
    never presented as feasible unless every target constraint actually passes.
    """
    if not isinstance(ar_id, str) or not ar_id:
        raise ValueError("ar_id must be a nonempty fixed candidate ID")
    if not _number(max_accuracy_gap) or not 0 <= max_accuracy_gap <= 1:
        raise ValueError("max_accuracy_gap must be a fraction between 0 and 1")
    if comparison_mode not in ("matched", "operating_point"):
        raise ValueError("comparison_mode must be matched or operating_point")
    rows = list(rows)
    ids = Counter(r.get("candidate_id") for r in rows if isinstance(r, dict)
                  and isinstance(r.get("candidate_id"), str))
    valid, rejected = [], []
    for index, row in enumerate(rows):
        reasons = _validate(row)
        if (isinstance(row, dict) and isinstance(row.get("candidate_id"), str)
                and ids.get(row["candidate_id"], 0) > 1):
            reasons.append("duplicate_candidate_id")
        if reasons:
            rejected.append({"row_index": index,
                             "candidate_id": row.get("candidate_id") if isinstance(row, dict) else None,
                             "failure_reasons": reasons})
        else:
            valid.append(row)
    report = {"schema_version": 1, "status": "no_feasible", "comparison_mode": comparison_mode,
              "fixed_ar_id": ar_id, "max_accuracy_gap": max_accuracy_gap,
              "target": {"accuracy": "AR > DARD > standard", "latency": "standard < DARD < AR",
                         "gap": "AR pass_rate - DARD pass_rate <= max_accuracy_gap"},
              "scope": "observed validation means; no global optimum or statistical guarantee",
              "relaxed_setting_keys": list(MATCHED_KEYS) if comparison_mode == "operating_point" else [],
              "ranking_policy": "feasible only: highest DARD accuracy, lowest DARD latency, highest standard accuracy",
              "native_reference_policy": "highest accuracy among valid same-context native candidates faster than DARD; weaker natives cannot establish feasibility",
              "fixed_ar": None, "feasible_triads": [], "best_gap_candidate": None,
              "triads": [], "rejected_rows": rejected, "failure_reasons": []}
    ar = next((r for r in valid if r["candidate_id"] == ar_id), None)
    if ar is None or ar["candidate"]["decoder"] != "ar":
        report.update(status="invalid_reference", failure_reasons=["fixed_ar_missing_invalid_or_not_ar"])
        return report
    report["fixed_ar"] = _point(ar)
    dards = [r for r in valid if r["candidate"]["decoder"] == "dard"]
    standards = [r for r in valid if r["candidate"]["decoder"] == "standard"]
    for label, values in (("dard", dards), ("standard", standards)):
        if not values:
            report["failure_reasons"].append("no_valid_" + label + "_candidates")
    for dard in dards:
        for standard in standards:
            reasons = []
            for label, row in (("dard", dard), ("standard", standard)):
                reasons.extend(label + "_vs_ar_mismatch:" + k for k in
                               _differences(row, ar, EVALUATION_KEYS))
                if row["expected_cases"] != ar["expected_cases"]:
                    reasons.append(label + "_vs_ar_mismatch:expected_cases")
            context_diff = _differences(dard, standard, MATCHED_KEYS)
            if comparison_mode == "matched":
                reasons.extend("dard_vs_standard_mismatch:" + k for k in context_diff)
            comparable = not reasons
            ap, dp, sp = ar["pass_rate"], dard["pass_rate"], standard["pass_rate"]
            at, dt, st = (x["mean_generation_seconds"] for x in (ar, dard, standard))
            gap = ap - dp
            constraints = {"ar_accuracy_gt_dard": ap > dp,
                           "dard_accuracy_gt_standard": dp > sp,
                           "standard_latency_lt_dard": st < dt,
                           "dard_latency_lt_ar": dt < at,
                           "accuracy_gap_within_limit": gap <= max_accuracy_gap or math.isclose(
                               gap, max_accuracy_gap, rel_tol=0, abs_tol=1e-12)}
            faster_native = [r for r in standards
                if r["mean_generation_seconds"] < dt
                and r["expected_cases"] == ar["expected_cases"]
                and not _differences(r, ar, EVALUATION_KEYS)
                and not _differences(r, standard, MATCHED_KEYS)]
            native_reference = min(faster_native,
                key=lambda r: (-r["pass_rate"], r["mean_generation_seconds"], r["candidate_id"])) if faster_native else None
            if comparable and native_reference is not None and sp < native_reference["pass_rate"]:
                reasons.append("standard_not_best_faster_native")
            reasons.extend(key for key, passed in constraints.items() if not passed)
            triad = {"ar_id": ar_id, "dard_id": dard["candidate_id"],
                     "standard_id": standard["candidate_id"], "comparable": comparable,
                     "feasible": not reasons, "failure_reasons": reasons,
                     "constraints": constraints, "accuracy_gap": gap,
                     "accuracy_gap_pp": 100 * gap,
                     "dard_pass_rate": dp, "standard_pass_rate": sp,
                     "ar_pass_rate": ap, "dard_mean_seconds": dt,
                     "standard_mean_seconds": st, "ar_mean_seconds": at,
                     "native_quality_reference": ({"candidate_id": native_reference["candidate_id"],
                         "pass_rate": native_reference["pass_rate"],
                         "mean_generation_seconds": native_reference["mean_generation_seconds"]}
                         if native_reference is not None else None),
                     "operating_point_differences": context_diff if comparison_mode == "operating_point" else []}
            report["triads"].append(triad)
    report["feasible_triads"] = sorted((t for t in report["triads"] if t["feasible"]),
        key=lambda t: (-t["dard_pass_rate"], t["dard_mean_seconds"], -t["standard_pass_rate"],
                       t["standard_mean_seconds"], t["dard_id"], t["standard_id"]))
    comparable = [t for t in report["triads"] if t["comparable"]]
    if comparable:
        report["best_gap_candidate"] = min(comparable,
            key=lambda t: (abs(t["accuracy_gap"]), len(t["failure_reasons"]),
                           t["dard_mean_seconds"], t["dard_id"], t["standard_id"]))
    if report["feasible_triads"]:
        report["status"] = "feasible"
    else:
        report["failure_reasons"].append("no_triad_satisfies_all_constraints")
    report["failure_reason_counts"] = dict(sorted(Counter(reason for t in report["triads"]
                                                       for reason in t["failure_reasons"]).items()))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="tradeoff_analysis.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ar-id", default=DEFAULT_AR_ID)
    parser.add_argument("--max-accuracy-gap", type=float, default=.05, help="fraction: .05 = 5 percentage points")
    parser.add_argument("--comparison-mode", choices=("matched", "operating_point"), default="matched")
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error("output must differ from input; source measurements are read-only")
    document = json.loads(args.input.read_text())
    if (not isinstance(document, dict) or document.get("selection_split") != "validation"
            or not isinstance(document.get("summaries"), list)):
        parser.error("input must be a validation tradeoff_analysis.json with normalized summaries")
    report = evaluate_target(document["summaries"], args.ar_id, args.max_accuracy_gap, args.comparison_mode)
    report["source_analysis"] = str(args.input.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "feasible_triads": len(report["feasible_triads"]),
                      "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
