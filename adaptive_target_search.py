"""Pure, bounded next-round search from measured code-generation summaries.

No model loading, inference, downloads or writes. Target attainment is a report
field, never a stop instruction. Accuracy/latency Pareto points remain available
even when every measured candidate misses the requested joint ordering.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import math

from dard_decoder import DARDConfig
from iterative_tuning import CPGRPO, INSTRUCT, DREAM_CODER, candidate, digest, validate_candidate
from target_tradeoff_search import DEFAULT_AR_ID, MATCHED_KEYS, EVALUATION_KEYS, _validate


HARD_CANDIDATE_CAP = 24
NATIVE_STEPS = (4, 8, 16, 256)


def _finite_fraction(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def _same(left, right, keys):
    a, b = left["comparable_settings"], right["comparable_settings"]
    return all((sorted(a[k]) if k == "seeds" else a[k]) ==
               (sorted(b[k]) if k == "seeds" else b[k]) for k in keys)


def _evaluation_matches(left, right, match_seeds):
    keys = EVALUATION_KEYS if match_seeds else tuple(k for k in EVALUATION_KEYS if k != "seeds")
    if not _same(left, right, keys):
        return False
    if match_seeds:
        return left["expected_cases"] == right["expected_cases"]
    # An explicitly provisional cross-seed screen still requires the same
    # number of tasks per seed; it never pools or adds measurements together.
    return (left["expected_cases"] * len(right["comparable_settings"]["seeds"]) ==
            right["expected_cases"] * len(left["comparable_settings"]["seeds"]))


def _point(row):
    return {key: deepcopy(row[key]) for key in ("candidate_id", "candidate", "pass_rate",
            "mean_generation_seconds", "expected_cases", "observed_cases", "passed",
            "comparable_settings")}


def score_candidates(rows, ar_id=DEFAULT_AR_ID, max_accuracy_gap=.05, *, match_seeds=True):
    """Rank closeness to the target while retaining the accuracy/time frontier.

    Distances are nonnegative and separately reported: accuracy gap beyond the
    allowed loss, native accuracy gain shortfall, and latency ratio violations.
    Ranking is lexicographic (quality gap, comparator availability, gain, speed),
    not a weighted sum that could hide a large accuracy deficit with speed.
    ``match_seeds=False`` permits clearly labelled provisional comparisons only;
    they cannot mark the target as established and never merge different seeds.
    """
    if not isinstance(ar_id, str) or not ar_id:
        raise ValueError("A fixed nonempty AR candidate ID is required")
    if not _finite_fraction(max_accuracy_gap):
        raise ValueError("max_accuracy_gap must be a finite fraction in [0, 1]")
    if type(match_seeds) is not bool:
        raise ValueError("match_seeds must be boolean")
    rows = list(rows)
    counts = Counter(r.get("candidate_id") for r in rows if isinstance(r, dict)
                     and isinstance(r.get("candidate_id"), str))
    good, rejected = [], []
    for index, row in enumerate(rows):
        errors = _validate(row)
        cid = row.get("candidate_id") if isinstance(row, dict) else None
        if isinstance(cid, str) and counts[cid] > 1:
            errors.append("duplicate_candidate_id")
        if not errors:
            try:
                validate_candidate(row["candidate"])
            except (ValueError, TypeError, KeyError):
                errors.append("invalid_candidate_config")
        if errors:
            rejected.append({"row_index": index, "candidate_id": cid, "reasons": errors})
        else:
            good.append(row)
    ar = next((r for r in good if r["candidate_id"] == ar_id and r["candidate"]["decoder"] == "ar"), None)
    report = {"schema_version": 1, "status": "ranked", "fixed_ar_id": ar_id,
        "max_accuracy_gap": max_accuracy_gap, "match_seeds": match_seeds,
        "search_should_continue": True, "target_met_count": 0,
        "ranking_policy": "lexicographic accuracy-gap excess, native availability, native-gain shortfall, AR accuracy order, DARD/AR latency excess, native/DARD latency excess; strict equalities are reported separately",
        "pareto_policy": "maximize observed DARD pass rate and minimize observed DARD mean generation seconds",
        "scope": "exploratory validation observations; no statistical or global-optimum claim",
        "fixed_ar": _point(ar) if ar else None, "ranked_candidates": [],
        "pareto_frontier": [], "rejected_rows": rejected}
    if ar is None:
        report["status"] = "invalid_reference"
        return report
    natives = [r for r in good if r["candidate"]["decoder"] == "standard"]
    for row in good:
        if row["candidate"]["decoder"] != "dard":
            continue
        if not _evaluation_matches(row, ar, match_seeds):
            rejected.append({"candidate_id": row["candidate_id"],
                             "reasons": ["evaluation_identity_differs_from_fixed_ar"]})
            continue
        peers = [n for n in natives if _same(row, n, MATCHED_KEYS)
                 and _evaluation_matches(row, n, match_seeds)]
        dt, dp = row["mean_generation_seconds"], row["pass_rate"]
        at, ap = ar["mean_generation_seconds"], ar["pass_rate"]
        faster = [n for n in peers if n["mean_generation_seconds"] < dt]
        if faster:
            native = min(faster, key=lambda n: (-n["pass_rate"], n["mean_generation_seconds"], n["candidate_id"]))
            reference_kind = "highest_accuracy_faster_native"
        elif peers:
            native = min(peers, key=lambda n: (n["mean_generation_seconds"], -n["pass_rate"], n["candidate_id"]))
            reference_kind = "fastest_native_but_none_faster_than_dard"
        else:
            native, reference_kind = None, "missing_native_comparator"
        np = native["pass_rate"] if native else None
        nt = native["mean_generation_seconds"] if native else None
        if not math.isfinite(dt / at) or (native is not None and not math.isfinite(nt / dt)):
            rejected.append({"candidate_id": row["candidate_id"], "reasons": ["nonfinite_latency_ratio"]})
            continue
        seed_matched = _same(row, ar, ("seeds",)) and (native is None or _same(row, native, ("seeds",)))
        distances = {
            "ar_accuracy_gap_excess": max(0., ap - dp - max_accuracy_gap),
            "native_accuracy_gain_shortfall": max(0., np - dp) if native else None,
            "ar_accuracy_order_shortfall": max(0., dp - ap),
            "native_over_dard_latency_excess": max(0., nt / dt - 1) if native else None,
            "dard_over_ar_latency_excess": max(0., dt / at - 1),
        }
        gap_within = ap - dp <= max_accuracy_gap or math.isclose(ap - dp, max_accuracy_gap, rel_tol=0, abs_tol=1e-12)
        constraints = {"ar_accuracy_gt_dard": ap > dp,
            "dard_accuracy_gt_native": dp > np if native else None,
            "ar_accuracy_gap_within_limit": gap_within,
            "native_latency_lt_dard": nt < dt if native else None,
            "dard_latency_lt_ar": dt < at, "same_evaluation_seeds": seed_matched}
        failed = [name for name, passed in constraints.items() if passed is not True]
        if native is None:
            failed.append("native_comparator_missing")
        scored = {**_point(row), "native_reference": _point(native) if native else None,
            "native_reference_kind": reference_kind, "native_comparator_count": len(peers),
            "faster_native_count": len(faster), "distances": distances,
            "accuracy_gap": ap - dp, "accuracy_gap_pp": 100 * (ap - dp),
            "native_accuracy_gain": dp - np if native else None,
            "native_over_dard_latency_ratio": nt / dt if native else None,
            "dard_over_ar_latency_ratio": dt / at,
            "constraints": constraints, "failure_reasons": failed,
            "target_met": not failed, "comparison_label": "same_seed_matched" if seed_matched else "provisional_cross_seed"}
        # Every component is finite. None values stay unknown in public output.
        scored["rank_vector"] = [distances["ar_accuracy_gap_excess"], int(native is None),
            distances["native_accuracy_gain_shortfall"] if native else 1.,
            distances["ar_accuracy_order_shortfall"],
            distances["dard_over_ar_latency_excess"],
            distances["native_over_dard_latency_excess"] if native else 1.]
        report["ranked_candidates"].append(scored)
    ranked = sorted(report["ranked_candidates"], key=lambda s: (tuple(s["rank_vector"]),
                    -s["pass_rate"], s["mean_generation_seconds"], s["candidate_id"]))
    report["ranked_candidates"] = ranked
    for row in ranked:
        dominated = any(other["pass_rate"] >= row["pass_rate"]
            and other["mean_generation_seconds"] <= row["mean_generation_seconds"]
            and (other["pass_rate"] > row["pass_rate"] or
                 other["mean_generation_seconds"] < row["mean_generation_seconds"])
            for other in ranked)
        if not dominated:
            report["pareto_frontier"].append(row)
    report["target_met_count"] = sum(r["target_met"] for r in ranked)
    return report


def _canonical(config):
    value = deepcopy(config)
    value.pop("id", None)
    value["forward_budget"] = 512
    value["dard"]["diagnostics"] = "summary"
    value["dard"]["max_step_records"] = 0
    validate_candidate(value)
    value["id"] = value["decoder"] + "-" + digest(value)[:16]
    return value


def _signature(config):
    """Behavior identity avoids duplicates from omitted defaults/adapter paths."""
    value = deepcopy(config)
    value.pop("id", None)
    adapter = value.get("adapter")
    value["adapter"] = adapter.get("sha256") if isinstance(adapter, dict) else None
    if value["decoder"] == "dard":
        value["dard"] = asdict(DARDConfig(**value["dard"]))
    else:
        value["dard"] = {k: value["dard"][k] for k in ("temperature", "top_p")}
    return digest(value)


def _native_bundle(dard):
    for steps in NATIVE_STEPS:
        c = candidate((dard["model"], dard["revision"]), decoder="standard",
            tokens=dard["tokens"], prefix=dard["prefix"], adapter=deepcopy(dard.get("adapter")),
            temperature=dard["dard"]["temperature"], top_p=dard["dard"]["top_p"],
            steps=steps, native_steps=steps, forward_budget=512)
        c["dard"].update(completion_policy="legacy_force", completion_verification_steps=2)
        yield _canonical(c)


def native_controls(config):
    """Return validated native 4/8/16/256-step controls for one DARD context.

    The caller should deduplicate old and new native controls by shared context
    and standard_steps, because older configurations may have different IDs for
    semantically irrelevant DARD metadata. Input is not changed.
    """
    if not isinstance(config, dict) or config.get("decoder") != "dard":
        raise ValueError("native_controls requires a DARD candidate")
    validate_candidate(config)
    return list(_native_bundle(config))


def joint_distance(point, accuracy_scale=.05):
    """Finite minimax violation for final diversity selection, or None if unknown.

    Accuracy violations are normalized by accuracy_scale; latency components
    are ratio excesses max(0, ratio-1). Zero does not certify strict orderings:
    equalities still fail the separately recorded target constraints. This
    helper does not change quality-first score_candidates ranking.
    """
    if (type(accuracy_scale) not in (int, float) or not math.isfinite(accuracy_scale)
            or not 0 < accuracy_scale <= 1):
        raise ValueError("accuracy_scale must be a finite positive fraction")
    if not isinstance(point, dict) or not isinstance(point.get("distances"), dict):
        raise ValueError("joint_distance requires a scored candidate point")
    distances = point["distances"]
    keys = ("ar_accuracy_gap_excess", "native_accuracy_gain_shortfall",
            "ar_accuracy_order_shortfall", "native_over_dard_latency_excess",
            "dard_over_ar_latency_excess")
    if any(k not in distances for k in keys):
        raise ValueError("Scored point lacks a required distance")
    if any(distances[k] is None for k in ("native_accuracy_gain_shortfall", "native_over_dard_latency_excess")):
        return None
    if any(type(distances[k]) not in (int, float) or not math.isfinite(distances[k])
           or distances[k] < 0 for k in keys):
        raise ValueError("Distances must be finite and nonnegative")
    value = max(distances[k] / accuracy_scale if "accuracy" in k else distances[k] for k in keys)
    if not math.isfinite(value):
        raise ValueError("Normalized joint distance overflowed")
    return value


def _neighbors(anchor, round_index):
    """Bounded coordinate neighbors plus quality-oriented combined proposals."""
    axes = [("tau_u", (.85, .9, .95, .97)), ("tau_c", (.4, .5, .6, .7)),
            ("steps_per_block", (32, 64, 128, 256)), ("transfer_limit", (4, 8, 16)),
            ("block_size", (64, 128, 256)), ("temperature", (0., .1, .3, 1.)),
            ("tokens", (256, 512)), ("top_p", (.9, .95, 1.)),
            ("completion_verification_steps", (2, 4, 8)),
            ("completion_policy", ("legacy_force", "force_then_verify", "verified_only"))]
    offset = round_index % len(axes)
    axes = axes[offset:] + axes[:offset]
    # Interleave axes instead of exhausting every value of the first coordinate.
    for index in range(max(len(values) for _, values in axes)):
        for key, values in axes:
            if index >= len(values):
                continue
            if key == "top_p" and anchor["dard"]["temperature"] == 0:
                continue
            c = deepcopy(anchor)
            if key == "tokens":
                c[key] = values[index]
            else:
                c["dard"][key] = values[index]
            if key == "temperature" and values[index] == 0:
                c["dard"]["top_p"] = 1.
            if key == "completion_verification_steps":
                c["dard"]["completion_policy"] = "force_then_verify"
            if c["dard"]["tau_c"] > c["dard"]["tau_u"]:
                continue
            yield c, "coordinate:" + key
        # Raising confidence, refinement budget and limiting transfer together
        # explores quality regions that single-coordinate changes can miss.
        if index < 4:
            c = deepcopy(anchor)
            c["dard"].update(tau_u=(.85, .9, .95, .97)[index],
                steps_per_block=(64, 128, 128, 256)[index], transfer_limit=(4, 8, 16, 4)[index])
            yield c, "combined_confidence_refinement_transfer"


def propose_next_candidates(rows, existing_candidates=(), ar_id=DEFAULT_AR_ID,
                            max_accuracy_gap=.05, *, round_index=0, max_candidates=24,
                            match_seeds=True, model_override=None):
    """Return <=24 deterministic, novel configs and required native controls.

    Each proposed DARD context has native 4/8/16/256-step comparators in this
    same batch, even if those controls ran in a previous round. This allows
    independent per-round evaluation without mixing repeated candidate/seeds.
    A bundle that does not fit is skipped intact. The fixed AR is not added;
    use max_candidates=23 if appending it under a 24-total run cap.
    ``model_override=INSTRUCT`` or ``DREAM_CODER`` permits explicit expansion to
    that pinned model. The caller must first admit the model using its verified
    preparation receipt. This pure function neither validates artifact receipts
    nor downloads models, and never chooses an unmeasured new model on its own.
    Existing DARD candidates must include scheduled as well as completed configs
    to avoid duplicate scheduling. Native controls intentionally repeat across
    rounds, once per context within a batch. No policy stops on target_met>0.
    """
    if type(round_index) is not int or round_index < 0:
        raise ValueError("round_index must be a nonnegative integer")
    if type(max_candidates) is not int or not 1 <= max_candidates <= HARD_CANDIDATE_CAP:
        raise ValueError("max_candidates must be between 1 and the hard cap 24")
    if model_override is not None:
        model_override = tuple(model_override)
        if model_override not in (CPGRPO, INSTRUCT, DREAM_CODER):
            raise ValueError("model_override must be a pinned CPGRPO, INSTRUCT or DREAM_CODER identity")
    rows = list(rows)
    scores = score_candidates(rows, ar_id, max_accuracy_gap, match_seeds=match_seeds)
    existing = [deepcopy(c) for c in existing_candidates]
    existing.extend(deepcopy(r["candidate"]) for r in rows
                    if isinstance(r, dict) and isinstance(r.get("candidate"), dict))
    seen, bad_existing = set(), []
    for c in existing:
        try:
            validate_candidate(c)
            seen.add(_signature(c))
        except (ValueError, TypeError, KeyError) as error:
            bad_existing.append({"candidate_id": c.get("id"), "reason": str(error)})
    # Mix quality-leading and Pareto-extreme anchors. Never discard a slow
    # high-quality point just because it currently violates the latency target.
    anchors, anchor_ids = [], set()
    scored = scores["ranked_candidates"]
    frontier = scores["pareto_frontier"]
    selected = scored[:2]
    if frontier:
        selected += [max(frontier, key=lambda r: r["pass_rate"]),
                     min(frontier, key=lambda r: r["mean_generation_seconds"])]
    for s in selected:
        if s["candidate_id"] not in anchor_ids:
            anchors.append(deepcopy(s["candidate"])); anchor_ids.add(s["candidate_id"])
    if not anchors:
        anchors = [candidate(CPGRPO, tokens=256, block=128, steps=64,
            temperature=0, top_p=1, tau_c=.6, tau_u=.8, prefix="none",
            native_steps=1, forward_budget=512)]
    for anchor in anchors:
        if model_override is not None and (anchor["model"], anchor["revision"]) != model_override:
            anchor["model"], anchor["revision"] = model_override
            anchor["adapter"] = None
        anchor["standard_steps"] = 1
        anchor["dard"].setdefault("completion_policy", "legacy_force")
        anchor["dard"].setdefault("completion_verification_steps", 2)
    pools = [list(_neighbors(anchor, round_index)) for anchor in anchors]
    proposals, provenance, skipped = [], [], 0
    batch_seen, repeated_control_ids = set(), []
    for position in range(max(len(pool) for pool in pools)):
        for anchor, pool in zip(anchors, pools):
            if position >= len(pool):
                continue
            raw, reason = pool[position]
            c = _canonical(raw)
            sig = _signature(c)
            if sig in seen:
                continue
            comparators = native_controls(c)
            new = [n for n in comparators if _signature(n) not in batch_seen] + [c]
            if len(proposals) + len(new) > max_candidates:
                skipped += 1
                continue
            for added in new:
                if added["decoder"] == "standard" and _signature(added) in seen:
                    repeated_control_ids.append(added["id"])
                batch_seen.add(_signature(added))
                seen.add(_signature(added)); proposals.append(added)
            provenance.append({"candidate_id": c["id"], "anchor_id": anchor.get("id"),
                "reason": reason, "native_comparator_ids": [n["id"] for n in comparators]})
    return {"schema_version": 1, "status": "proposed" if proposals else "no_novel_bundle_fits",
        "round_index": round_index, "fixed_ar_id": ar_id, "max_accuracy_gap": max_accuracy_gap,
        "candidate_cap": max_candidates, "hard_candidate_cap": HARD_CANDIDATE_CAP,
        "forward_budget": 512, "model_override": list(model_override) if model_override else None,
        "model_admission_policy": "caller must verify model preparation receipt before supplying a new-model override",
        "candidates": proposals, "provenance": provenance,
        "control_policy": "all native controls repeated in every batch; analyze each round independently",
        "fixed_ar_included": False, "repeated_control_ids": repeated_control_ids,
        "skipped_bundles_for_cap": skipped, "invalid_existing_candidates": bad_existing,
        "scores": scores, "search_should_continue": True}
