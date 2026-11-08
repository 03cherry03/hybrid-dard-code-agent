"""Adaptive search keeps difficult targets honest and each GPU round bounded."""
from copy import deepcopy
import json

import pytest

from adaptive_target_search import (score_candidates, propose_next_candidates, _signature,
                                    NATIVE_STEPS, native_controls, joint_distance)
from iterative_tuning import AR, CPGRPO, INSTRUCT, DREAM_CODER, candidate, digest, validate_candidate


def row(cid, decoder, passed, seconds, *, seeds=(1,), **updates):
    cfg = candidate(AR if decoder == "ar" else CPGRPO, decoder=decoder,
        tokens=256, temperature=0, top_p=1, prefix="none", native_steps=1 if decoder == "dard" else 16,
        forward_budget=512, steps=64, tau_c=.6, tau_u=.8)
    cfg.update(updates); cfg["id"] = cid
    setting = dict(model=cfg["model"], revision=cfg["revision"], adapter_sha256=None,
        tokens=cfg["tokens"], prefix=cfg["prefix"], temperature=cfg["dard"]["temperature"],
        top_p=cfg["dard"]["top_p"], taskset_sha256="tasks", seeds=list(seeds),
        evaluation_protocol=dict(candidate_warmup=True, counterbalanced_order=True,
            extraction_version="v2", generation_timeout=120, grading_timeout=300))
    n = 100 * len(seeds)
    return dict(candidate_id=cid, candidate=cfg, comparable_settings=setting,
        expected_cases=n, observed_cases=n, passed=passed * len(seeds), pass_rate=passed / 100,
        mean_generation_seconds=seconds, valid_for_selection=True,
        missing_or_invalid_cases=0, infrastructure_error_count=0)


def measurements():
    return [row("ar", "ar", 80, 3), row("quality", "dard", 77, 5),
            row("fast", "dard", 60, 1), row("native", "standard", 50, .5)]


def test_distances_do_not_hide_large_accuracy_gap_with_speed():
    report = score_candidates(measurements(), "ar")
    assert report["ranked_candidates"][0]["candidate_id"] == "quality"
    assert report["ranked_candidates"][0]["distances"]["ar_accuracy_gap_excess"] == 0
    assert report["ranked_candidates"][0]["distances"]["dard_over_ar_latency_excess"] > 0
    assert report["ranked_candidates"][1]["distances"]["ar_accuracy_gap_excess"] == pytest.approx(.15)
    assert report["search_should_continue"] is True


def test_pareto_keeps_slow_quality_and_fast_lower_quality_but_removes_dominated():
    values = measurements() + [row("dominated", "dard", 59, 2)]
    report = score_candidates(values, "ar")
    assert {r["candidate_id"] for r in report["pareto_frontier"]} == {"quality", "fast"}


def test_strongest_faster_native_prevents_cherry_picking():
    values = [row("ar", "ar", 80, 3), row("dard", "dard", 77, 2),
              row("weak", "standard", 50, .5), row("strong", "standard", 78, 1.5)]
    r = score_candidates(values, "ar")["ranked_candidates"][0]
    assert r["native_reference"]["candidate_id"] == "strong"
    assert r["distances"]["native_accuracy_gain_shortfall"] == pytest.approx(.01)
    assert r["target_met"] is False


def test_none_faster_and_missing_native_remain_explicit():
    values = [row("ar", "ar", 80, 3), row("dard", "dard", 77, 1), row("slow", "standard", 50, 2)]
    r = score_candidates(values, "ar")["ranked_candidates"][0]
    assert r["native_reference_kind"] == "fastest_native_but_none_faster_than_dard"
    assert r["distances"]["native_over_dard_latency_excess"] >= 1
    missing = score_candidates(values[:2], "ar")["ranked_candidates"][0]
    assert missing["native_reference"] is None
    assert missing["distances"]["native_accuracy_gain_shortfall"] is None
    assert not missing["target_met"]


def test_target_success_does_not_stop_search():
    values = [row("ar", "ar", 80, 3), row("dard", "dard", 77, 2), row("s", "standard", 70, 1)]
    r = score_candidates(values, "ar")
    assert r["target_met_count"] == 1 and r["search_should_continue"]
    p = propose_next_candidates(values, ar_id="ar")
    assert p["candidates"] and p["search_should_continue"]


def test_equal_orderings_are_not_target_success_and_distances_nonnegative():
    values = [row("ar", "ar", 80, 2), row("dard", "dard", 80, 2), row("s", "standard", 80, 2)]
    r = score_candidates(values, "ar")["ranked_candidates"][0]
    assert not r["target_met"]
    assert all(v >= 0 for v in r["distances"].values())
    assert r["distances"]["ar_accuracy_order_shortfall"] == 0
    assert r["constraints"]["ar_accuracy_gt_dard"] is False


def test_fixed_ar_is_not_replaced_by_easier_reference():
    values = measurements() + [row("weak_ar", "ar", 61, 4)]
    report = score_candidates(values, "ar")
    assert report["fixed_ar"]["pass_rate"] == .8
    assert score_candidates(values, "missing")["status"] == "invalid_reference"


def test_seed_relaxation_is_explicit_provisional_and_never_pools():
    values = [row("ar", "ar", 80, 3, seeds=(1, 2, 3)),
              row("dard", "dard", 77, 2), row("s", "standard", 70, 1)]
    assert not score_candidates(values, "ar")["ranked_candidates"]
    relaxed = score_candidates(values, "ar", match_seeds=False)["ranked_candidates"][0]
    assert relaxed["comparison_label"] == "provisional_cross_seed"
    assert relaxed["observed_cases"] == 100 and relaxed["target_met"] is False


def test_taskset_or_protocol_mismatch_cannot_be_relaxed():
    values = measurements(); values[1]["comparable_settings"]["taskset_sha256"] = "different"
    report = score_candidates(values, "ar", match_seeds=False)
    assert "quality" not in {r["candidate_id"] for r in report["ranked_candidates"]}


def test_invalid_measurements_do_not_create_nan_scores():
    values = measurements(); values[1]["pass_rate"] = float("nan")
    report = score_candidates(values, "ar")
    json.dumps(report, allow_nan=False)
    assert "quality" not in {r["candidate_id"] for r in report["ranked_candidates"]}


def test_extreme_finite_times_do_not_overflow_output_ratios():
    values = [row("ar", "ar", 80, 1e-300), row("dard", "dard", 77, 1e300),
              row("native", "standard", 50, 1)]
    report = score_candidates(values, "ar")
    assert not report["ranked_candidates"]
    assert report["rejected_rows"][-1]["reasons"] == ["nonfinite_latency_ratio"]
    json.dumps(report, allow_nan=False)


def test_score_rejects_invalid_model_config_before_neighbor_generation():
    values = measurements(); values[1]["candidate"]["dard"]["tau_u"] = float("nan")
    report = score_candidates(values, "ar")
    assert "quality" not in {r["candidate_id"] for r in report["ranked_candidates"]}
    proposals = propose_next_candidates(values, ar_id="ar")
    assert proposals["candidates"]
    json.dumps(proposals, allow_nan=False)


def test_proposals_are_deterministic_bounded_valid_and_source_unchanged():
    values = measurements(); before = deepcopy(values)
    first = propose_next_candidates(values, ar_id="ar", round_index=2)
    second = propose_next_candidates(values, ar_id="ar", round_index=2)
    assert first == second and values == before
    assert 0 < len(first["candidates"]) <= 24
    assert len({_signature(c) for c in first["candidates"]}) == len(first["candidates"])
    for c in first["candidates"]:
        validate_candidate(c)
        assert c["forward_budget"] == 512
        body = {k: v for k, v in c.items() if k != "id"}
        assert c["id"] == c["decoder"] + "-" + digest(body)[:16]
    json.dumps(first, allow_nan=False)


def context(c):
    return (c["model"], c["revision"], c["tokens"], c["prefix"],
            c["dard"]["temperature"], c["dard"]["top_p"], digest(c.get("adapter")))


def test_every_new_context_has_fast_and_quality_native_comparators():
    result = propose_next_candidates(measurements(), ar_id="ar")
    for c in result["candidates"]:
        if c["decoder"] == "dard":
            steps = {n["standard_steps"] for n in result["candidates"]
                     if n["decoder"] == "standard" and context(n) == context(c)}
            assert set(NATIVE_STEPS) <= steps


def test_next_round_does_not_reschedule_dard_but_repeats_controls():
    first = propose_next_candidates(measurements(), ar_id="ar")
    second = propose_next_candidates(measurements(), first["candidates"], ar_id="ar", round_index=1)
    assert not {_signature(c) for c in first["candidates"] if c["decoder"] == "dard"} & {
        _signature(c) for c in second["candidates"] if c["decoder"] == "dard"}
    assert second["repeated_control_ids"]
    for c in second["candidates"]:
        if c["decoder"] == "dard":
            assert set(NATIVE_STEPS) <= {n["standard_steps"] for n in second["candidates"]
                if n["decoder"] == "standard" and context(n) == context(c)}


def test_small_cap_never_schedules_partial_new_context_bundle():
    # No existing native means four controls plus one DARD are indivisible.
    values = measurements()[:3]
    report = propose_next_candidates(values, ar_id="ar", max_candidates=4)
    assert report["candidates"] == []
    assert report["status"] == "no_novel_bundle_fits"


def test_instruct_expansion_is_explicit_pinned_and_has_no_cpgrpo_adapter():
    result = propose_next_candidates(measurements(), ar_id="ar", model_override=INSTRUCT)
    assert result["candidates"]
    assert all((c["model"], c["revision"]) == INSTRUCT and c["adapter"] is None for c in result["candidates"])
    with pytest.raises(ValueError, match="pinned"):
        propose_next_candidates(measurements(), ar_id="ar", model_override=("unknown", "main"))


def test_dream_expansion_requires_explicit_caller_override_after_its_receipt_gate():
    # Receipt inspection/admission belongs to the controller. The pure proposer
    # must not opportunistically switch checkpoints while that gate is pending.
    without_override = propose_next_candidates(measurements(), ar_id="ar")
    assert without_override["model_override"] is None
    assert all((c["model"], c["revision"]) == CPGRPO for c in without_override["candidates"])
    admitted = propose_next_candidates(measurements(), ar_id="ar", model_override=DREAM_CODER)
    assert admitted["candidates"] and admitted["model_override"] == list(DREAM_CODER)
    assert "caller must verify" in admitted["model_admission_policy"]
    for c in admitted["candidates"]:
        validate_candidate(c)
        assert (c["model"], c["revision"]) == DREAM_CODER
        assert c["adapter"] is None and c["forward_budget"] == 512
    with pytest.raises(ValueError, match="pinned"):
        propose_next_candidates(measurements(), ar_id="ar", model_override=(DREAM_CODER[0], "main"))


def test_full_pareto_is_returned_even_when_more_than_anchor_budget():
    values = [row("ar", "ar", 90, 3), row("native", "standard", 30, .1)]
    values += [row("d" + str(i), "dard", 40 + i * 5, .5 + i) for i in range(7)]
    report = score_candidates(values, "ar")
    assert len(report["pareto_frontier"]) == 7
    assert report["search_should_continue"] is True


def test_public_native_controls_complete_context_and_do_not_change_anchor():
    anchor = measurements()[1]["candidate"]
    before = deepcopy(anchor)
    controls = native_controls(anchor)
    assert anchor == before
    assert {c["standard_steps"] for c in controls} == set(NATIVE_STEPS)
    assert all(context(c) == context(anchor) and c["decoder"] == "standard" for c in controls)
    for c in controls:
        validate_candidate(c)
    with pytest.raises(ValueError, match="DARD"):
        native_controls(measurements()[0]["candidate"])


def test_joint_distance_can_preserve_balanced_point_alongside_slow_quality_leader():
    values = [row("ar", "ar", 80, 3), row("quality", "dard", 79, 30),
              row("balanced", "dard", 70, 2), row("native", "standard", 60, 1)]
    scores = score_candidates(values, "ar")["ranked_candidates"]
    assert scores[0]["candidate_id"] == "quality"  # Existing ranking unchanged.
    selected = min(scores, key=joint_distance)
    assert selected["candidate_id"] == "balanced"
    assert joint_distance(selected) == pytest.approx(1)


def test_joint_distance_uses_excess_ratios_and_strict_flags_remain_separate():
    values = [row("ar", "ar", 80, 3), row("d", "dard", 77, 2), row("s", "standard", 70, 1)]
    p = score_candidates(values, "ar")["ranked_candidates"][0]
    assert p["target_met"] and joint_distance(p) == 0
    values[2]["mean_generation_seconds"] = 2
    p = score_candidates(values, "ar")["ranked_candidates"][0]
    assert not p["target_met"] and joint_distance(p) == 0


def test_joint_distance_does_not_invent_missing_native_measurements():
    p = score_candidates(measurements()[:3], "ar")["ranked_candidates"][0]
    assert joint_distance(p) is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -.1])
def test_joint_distance_rejects_invalid_score_values(bad):
    p = score_candidates(measurements(), "ar")["ranked_candidates"][0]
    p["distances"]["ar_accuracy_gap_excess"] = bad
    with pytest.raises(ValueError, match="finite"):
        joint_distance(p)


@pytest.mark.parametrize("cap", [0, 25, True, 3.5])
def test_hard_round_cap(cap):
    with pytest.raises(ValueError):
        propose_next_candidates(measurements(), ar_id="ar", max_candidates=cap)
