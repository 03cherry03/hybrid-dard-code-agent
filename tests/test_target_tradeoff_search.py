"""Reject tempting but incomparable or target-violating measured configurations."""
from copy import deepcopy
import json

import pytest

from target_tradeoff_search import evaluate_target, main


def row(cid, decoder, passed, seconds, **settings):
    comparable = dict(model="AR" if decoder == "ar" else "diffusion", revision="fixed",
                      adapter_sha256=None, tokens=256, prefix="none", temperature=0, top_p=1,
                      taskset_sha256="fixed-tasks", seeds=[1, 2, 3],
                      evaluation_protocol=dict(candidate_warmup=True, counterbalanced_order=True,
                          extraction_version="v2", generation_timeout=120, grading_timeout=300))
    comparable.update(settings)
    candidate = {k: comparable[k] for k in ("model", "revision", "tokens", "prefix")}
    candidate.update(id=cid, decoder=decoder, adapter=None,
                     dard={k: comparable[k] for k in ("temperature", "top_p")})
    return dict(candidate_id=cid, candidate=candidate, comparable_settings=comparable,
                valid_for_selection=True, expected_cases=100, observed_cases=100, passed=passed,
                pass_rate=passed / 100, mean_generation_seconds=seconds,
                missing_or_invalid_cases=0, infrastructure_error_count=0)


def rows():
    return [row("ar", "ar", 80, 3), row("d", "dard", 77, 2), row("s", "standard", 70, 1)]


def test_inclusive_three_point_gap_and_strict_joint_ordering():
    before = rows()
    result = evaluate_target(before, "ar", max_accuracy_gap=.03)
    assert result["status"] == "feasible"
    assert result["feasible_triads"][0]["accuracy_gap_pp"] == pytest.approx(3)
    assert before == rows()  # Pure: measurements and metadata are not changed.


@pytest.mark.parametrize("index,changes,reason", [
    (1, {"passed": 76, "pass_rate": .76}, "accuracy_gap_within_limit"),
    (1, {"passed": 80, "pass_rate": .80}, "ar_accuracy_gt_dard"),
    (2, {"passed": 77, "pass_rate": .77}, "dard_accuracy_gt_standard"),
    (1, {"mean_generation_seconds": 1}, "standard_latency_lt_dard"),
    (1, {"mean_generation_seconds": 3}, "dard_latency_lt_ar"),
    (2, {"mean_generation_seconds": 4}, "standard_latency_lt_dard"),
])
def test_each_target_constraint_is_enforced(index, changes, reason):
    values = rows(); values[index].update(changes)
    result = evaluate_target(values, "ar", max_accuracy_gap=.03)
    assert result["status"] == "no_feasible"
    assert reason in result["triads"][0]["failure_reasons"]


def test_all_failed_constraints_are_reported_together():
    values = [row("ar", "ar", 80, 1), row("d", "dard", 30, 2), row("s", "standard", 50, 3)]
    result = evaluate_target(values, "ar")
    assert set(result["triads"][0]["failure_reasons"]) == {
        "accuracy_gap_within_limit", "dard_accuracy_gt_standard",
        "standard_latency_lt_dard", "dard_latency_lt_ar"}
    assert result["best_gap_candidate"]["feasible"] is False


def test_default_gap_is_five_points_and_boundary_is_inclusive():
    values = rows(); values[1].update(passed=75, pass_rate=.75)
    assert evaluate_target(values, "ar")["status"] == "feasible"
    assert evaluate_target(values, "ar", .03)["status"] == "no_feasible"
    values[1].update(passed=74, pass_rate=.74)
    assert evaluate_target(values, "ar")["status"] == "no_feasible"


def test_weak_native_cannot_manufacture_target_when_stronger_faster_native_exists():
    values = rows() + [row("s-strong", "standard", 78, 1.5)]
    result = evaluate_target(values, "ar")
    assert result["status"] == "no_feasible"
    weak = next(t for t in result["triads"] if t["standard_id"] == "s")
    assert "standard_not_best_faster_native" in weak["failure_reasons"]
    assert weak["native_quality_reference"]["candidate_id"] == "s-strong"
    strong = next(t for t in result["triads"] if t["standard_id"] == "s-strong")
    assert "dard_accuracy_gt_standard" in strong["failure_reasons"]


def test_native_reference_must_really_be_faster_and_match_context():
    values = rows() + [row("slow-strong", "standard", 79, 2.1),
                      row("different-strong", "standard", 79, 1.5, temperature=.3)]
    result = evaluate_target(values, "ar")
    assert result["status"] == "feasible"
    assert result["feasible_triads"][0]["standard_id"] == "s"


@pytest.mark.parametrize("change,reason", [
    ({"valid_for_selection": False}, "not_valid_for_selection"),
    ({"observed_cases": 99}, "incomplete_case_coverage"),
    ({"missing_or_invalid_cases": 1}, "missing_or_invalid_cases_not_zero"),
    ({"infrastructure_error_count": 1}, "infrastructure_error_count_not_zero"),
    ({"pass_rate": float("nan")}, "invalid_pass_rate"),
    ({"mean_generation_seconds": float("inf")}, "invalid_mean_generation_seconds"),
    ({"pass_rate": .78}, "pass_rate_count_mismatch"),
    ({"split": "holdout"}, "not_validation"),
    ({"comparable_settings": {}}, "incomplete_evaluation_protocol"),
])
def test_invalid_or_incomplete_rows_cannot_satisfy_target(change, reason):
    values = rows(); values[1].update(change)
    result = evaluate_target(values, "ar")
    assert result["status"] == "no_feasible"
    assert reason in result["rejected_rows"][0]["failure_reasons"]
    assert result["best_gap_candidate"] is None


@pytest.mark.parametrize("field,value", [("seeds", [1]), ("taskset_sha256", "other-tasks"),
    ("evaluation_protocol", dict(candidate_warmup=True, counterbalanced_order=True,
        extraction_version="v3", generation_timeout=120, grading_timeout=300))])
@pytest.mark.parametrize("mode", ["matched", "operating_point"])
def test_seed_taskset_protocol_never_relaxed(field, value, mode):
    values = rows(); values[1]["comparable_settings"][field] = value
    result = evaluate_target(values, "ar", comparison_mode=mode)
    assert result["status"] == "no_feasible"
    assert "dard_vs_ar_mismatch:" + field in result["triads"][0]["failure_reasons"]
    assert result["best_gap_candidate"] is None


@pytest.mark.parametrize("field,value", [("model", "other"), ("revision", "other"),
    ("tokens", 512), ("prefix", "code_fence"), ("temperature", .3), ("top_p", .95)])
def test_decoder_context_is_matched_unless_explicit_operating_point(field, value):
    values = [row("ar", "ar", 80, 3), row("d", "dard", 77, 2, **{field: value}),
              row("s", "standard", 70, 1)]
    matched = evaluate_target(values, "ar")
    assert matched["status"] == "no_feasible"
    assert "dard_vs_standard_mismatch:" + field in matched["triads"][0]["failure_reasons"]
    relaxed = evaluate_target(values, "ar", comparison_mode="operating_point")
    assert relaxed["status"] == "feasible"
    assert relaxed["triads"][0]["operating_point_differences"] == [field]


def test_ar_is_fixed_not_chosen_to_manufacture_small_gap():
    values = rows(); values[0].update(passed=90, pass_rate=.9)
    values.append(row("weak-ar", "ar", 80, 3))
    result = evaluate_target(values, "ar")
    assert result["status"] == "no_feasible"
    assert result["fixed_ar"]["pass_rate"] == .9
    assert evaluate_target(values, "missing")["status"] == "invalid_reference"
    assert evaluate_target(values, "s")["status"] == "invalid_reference"


def test_duplicate_id_rejected_instead_of_mixing_measurements():
    values = rows(); copy = deepcopy(values[1]); copy["comparable_settings"]["seeds"] = [1]
    values.append(copy)
    result = evaluate_target(values, "ar")
    assert result["status"] == "no_feasible"
    assert len(result["rejected_rows"]) == 2
    assert all("duplicate_candidate_id" in r["failure_reasons"] for r in result["rejected_rows"])


@pytest.mark.parametrize("change,reason", [
    ({"candidate_id": []}, "missing_candidate_id"),
    ({"seeds": 1}, "seed_metadata_mismatch"),
])
def test_malformed_metadata_is_rejected_without_crashing(change, reason):
    values = rows(); values[1].update(change)
    result = evaluate_target(values, "ar")
    assert result["status"] == "no_feasible"
    assert reason in result["rejected_rows"][0]["failure_reasons"]


def test_incomplete_protocol_values_cannot_be_matched_by_shared_nulls():
    values = rows()
    for value in values:
        value["comparable_settings"]["evaluation_protocol"]["generation_timeout"] = None
    result = evaluate_target(values, "ar")
    assert result["status"] == "invalid_reference"
    assert all("invalid_evaluation_protocol" in r["failure_reasons"] for r in result["rejected_rows"])


def test_adapter_weight_identity_is_required_and_matched():
    values = rows(); hashes = {"adapter_model.safetensors": "a" * 64}
    values[1]["candidate"]["adapter"] = {"sha256": hashes}
    values[1]["comparable_settings"]["adapter_sha256"] = hashes
    result = evaluate_target(values, "ar")
    assert "dard_vs_standard_mismatch:adapter_sha256" in result["triads"][0]["failure_reasons"]


def test_feasible_ranking_retains_quality_first_and_marks_closest_failure():
    values = rows() + [row("d-better", "dard", 79, 2.5), row("d-too-slow", "dard", 80, 4)]
    result = evaluate_target(values, "ar")
    assert [t["dard_id"] for t in result["feasible_triads"]] == ["d-better", "d"]
    assert result["best_gap_candidate"]["dard_id"] == "d-too-slow"
    assert not result["best_gap_candidate"]["feasible"]


def test_cli_requires_validation_report_and_preserves_source(tmp_path):
    source = tmp_path / "input.json"; out = tmp_path / "out.json"
    source.write_text(json.dumps({"selection_split": "validation", "summaries": rows()}))
    original = source.read_bytes()
    assert main(["--input", str(source), "--output", str(out), "--ar-id", "ar"]) == 0
    assert json.loads(out.read_text())["status"] == "feasible"
    assert source.read_bytes() == original
    with pytest.raises(SystemExit):
        main(["--input", str(source), "--output", str(source)])
    source.write_text(json.dumps({"selection_split": "holdout", "summaries": rows()}))
    with pytest.raises(SystemExit):
        main(["--input", str(source), "--output", str(out)])


@pytest.mark.parametrize("gap", [-.01, 3, float("nan"), True])
def test_gap_is_a_finite_fraction(gap):
    with pytest.raises(ValueError):
        evaluate_target(rows(), "ar", gap)
