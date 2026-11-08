#!/usr/bin/env python3
"""Candidate design for a DARD repair campaign.

The campaign is deliberately staged: a small development cohort screens
completion and convergence behaviour, a full validation cohort compares only
the surviving contexts, and fresh seeds are reserved for the final shortlist.
This module only creates frozen configurations and selection metadata; it does
not run inference or use scores to alter the task manifests.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iterative_tuning import AR, CPGRPO, candidate, digest, validate_candidate
from scripts.tradeoff_experiments import configured, shared_context


DEV_TASKS_PER_LANGUAGE = 12
DEV_LANGUAGES = ("py", "js", "cpp", "rs")
NATIVE_STEPS = (4, 8, 16, 32)
FINAL_NATIVE_STEPS = (64, 128)
COMPLETION_POLICIES = ("resample_then_verify", "verified_only", "legacy_force")


@dataclass(frozen=True)
class RepairCampaign:
    schema_version: int
    name: str
    model: tuple[str, str]
    max_accuracy_gap: float
    dev_tasks_per_language: int
    final_seed_ids: tuple[int, ...]
    grading_workers: int
    grading_queue_size: int
    no_holdout_selection: bool = True

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "model": list(self.model),
            "max_accuracy_gap": self.max_accuracy_gap,
            "dev_tasks_per_language": self.dev_tasks_per_language,
            "dev_languages": list(DEV_LANGUAGES),
            "final_seed_ids": list(self.final_seed_ids),
            "grading_workers": self.grading_workers,
            "grading_queue_size": self.grading_queue_size,
            "no_holdout_selection": self.no_holdout_selection,
            "selection": {
                "primary": "minimize joint target violation after certified completion gate",
                "quality": "functional pass rate, certified pass rate, and forced case rate are all retained",
                "speed": "mean and p95 generation seconds; DARD must be compared with same-context native controls",
                "strict_target": "AR > DARD > native and native_time < DARD_time < AR_time",
                "target_gap_pp": 5,
                "legacy_force": "diagnostic only; never sole final-quality evidence",
            },
            "model_admission": {
                "base": "cpGRPO",
                "dream": "separate native/block64 canary, then DARD canary; OOM quarantines only the failed configuration",
                "quantization": "bf16 and 4bit are separate timing protocols and are never pooled",
            },
        }


def _with_policy(config, policy, verification_steps=2):
    value = deepcopy(config)
    value["dard"]["completion_policy"] = policy
    value["dard"]["completion_verification_steps"] = verification_steps
    value.pop("id", None)
    value["id"] = value["decoder"] + "-" + digest(value)[:16]
    validate_candidate(value)
    return value


def _native_bundle(model, tokens, prefix, temperature, top_p):
    values = []
    for steps in NATIVE_STEPS:
        value = configured(decoder="standard", tokens=tokens, prefix=prefix,
                           temperature=temperature, top_p=top_p, steps=steps,
                           block=128)
        value["model"], value["revision"] = model
        value.pop("id", None)
        value["id"] = value["decoder"] + "-" + digest(value)[:16]
        validate_candidate(value)
        values.append(value)
    return values


def _dard_bundle(tokens, prefix, temperature, top_p):
    """Thirteen DARD configs covering convergence repair and speed controls."""
    specs = [
        # The primary repair path: fresh masked sampling after a cycle.
        (128, 32, .4, .80, 4, "resample_then_verify", 2),
        (128, 48, .4, .80, 4, "resample_then_verify", 2),
        (128, 64, .4, .80, 4, "resample_then_verify", 2),
        (128, 48, .5, .80, 2, "resample_then_verify", 2),
        (128, 64, .5, .85, 4, "resample_then_verify", 4),
        (64, 48, .4, .80, 4, "resample_then_verify", 2),
        (128, 64, .4, .75, 8, "resample_then_verify", 2),
        (128, 64, .6, .85, 4, "resample_then_verify", 4),
        # Strict completion controls expose the quality cost of forcing.
        (128, 32, .4, .80, 4, "verified_only", 2),
        (128, 64, .4, .80, 4, "verified_only", 4),
        (64, 48, .5, .80, 4, "verified_only", 2),
        (128, 64, .6, .85, 4, "verified_only", 4),
        # One historical control per context is sufficient; the completed
        # predecessor campaign already contains the other legacy variants.
        (128, 64, .4, .80, 4, "legacy_force", 2),
    ]
    result = []
    for block, steps, tau_c, tau_u, transfer, policy, verify_steps in specs:
        base = configured(tokens=tokens, prefix=prefix, temperature=temperature,
                          top_p=top_p, block=block, steps=steps, tau_c=tau_c,
                          tau_u=tau_u)
        base["dard"]["transfer_limit"] = transfer
        result.append(_with_policy(base, policy, verify_steps))
    return result


def repair_grid(model=CPGRPO):
    """Return AR, native controls, and DARD repair candidates.

    Every DARD point has four native controls with identical model, token
    budget, prefix, temperature, and top-p.  The returned IDs are unique.
    """
    configs = {}
    for tokens, prefix, temperature, top_p in (
            (256, "none", 0., 1.), (256, "code_fence", 0., 1.),
            (512, "none", 0., 1.), (512, "code_fence", 0., 1.)):
        for value in _native_bundle(model, tokens, prefix, temperature, top_p):
            configs[value["id"]] = value
        for value in _dard_bundle(tokens, prefix, temperature, top_p):
            value["model"], value["revision"] = model
            value.pop("id", None)
            value["id"] = value["decoder"] + "-" + digest(value)[:16]
            validate_candidate(value)
            configs[value["id"]] = value
    for tokens in (256, 512):
        value = candidate(AR, decoder="ar", tokens=tokens, native_steps=1,
                          temperature=0, top_p=1, forward_budget=512,
                          prefix="code_fence")
        configs[value["id"]] = value
    values = list(configs.values())
    if len({value["id"] for value in values}) != len(values):
        raise AssertionError("Repair grid contains duplicate candidate IDs")
    for dard in (value for value in values if value["decoder"] == "dard"):
        matched = [value for value in values if value["decoder"] == "standard"
                   and shared_context(value) == shared_context(dard)]
        if {value["standard_steps"] for value in matched} != set(NATIVE_STEPS):
            raise AssertionError("DARD context lacks the complete native curve")
    return values


def campaign_metadata(model=CPGRPO):
    campaign = RepairCampaign(1, "dard_completion_speed_repair_v1", model,
                              .05, DEV_TASKS_PER_LANGUAGE,
                              (20261111, 20261112, 20261113), 8, 64)
    metadata = campaign.to_dict()
    metadata.update({"candidate_count": len(repair_grid(model)),
                     "final_native_steps": list(FINAL_NATIVE_STEPS),
                     "native_steps": list(NATIVE_STEPS),
                     "completion_policies": list(COMPLETION_POLICIES),
                     "phase_order": [
                         "dev_48_tasks_completion_and_cycle_screen",
                         "validation_top_4_same_context_native_curves",
                         "final_three_fresh_seeds",
                     ],
                     "selection_exclusions": [
                         "holdout data",
                         "incomplete candidates",
                         "legacy_force-only quality claims",
                         "mixed bf16/4bit latency rows",
                         "mixed prefix or output-length contexts",
                     ]})
    return metadata


def select_dev_rows(rows, max_candidates=8):
    """Choose validation candidates from complete dev rows without pooling seeds."""
    if type(max_candidates) is not int or not 1 <= max_candidates <= 16:
        raise ValueError("max_candidates must be between 1 and 16")
    valid = [row for row in rows if row.get("valid_for_selection") is True
             and row.get("missing_or_invalid_cases", 0) == 0
             and row.get("infrastructure_error_count", 0) == 0
             and row.get("candidate", {}).get("dard", {}).get("completion_policy") != "legacy_force"]
    # Keep the best certified/functional point and the fastest points near it;
    # the caller still repeats every native control for selected contexts.
    def key(row):
        certified = row.get("certified_pass_rate", row.get("completion_qualified_pass_rate", 0.))
        forced = row.get("forced_case_rate", 1.)
        return (-float(row.get("pass_rate", 0.)), -float(certified),
                float(forced), float(row.get("mean_generation_seconds", float("inf"))))
    result = []
    for row in sorted(valid, key=key):
        if row["candidate"]["decoder"] != "dard":
            continue
        if row["candidate"]["id"] not in {r["candidate"]["id"] for r in result}:
            result.append(row)
        if len(result) == max_candidates:
            break
    return result


if __name__ == "__main__":
    import json
    print(json.dumps(campaign_metadata(), indent=2, ensure_ascii=False))
