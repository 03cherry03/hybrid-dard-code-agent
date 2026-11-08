"""Source-paired validation selection, honest failure denominators, and exports."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

import tradeoff_analysis as analysis
from iterative_tuning import digest, file_sha, task_id


def config(cid, *, decoder="standard", **updates):
    value = {"id": cid, "decoder": decoder, "model": "fixture/model", "revision": "pinned-revision",
             "adapter": None, "prefix": "code_fence", "tokens": 512,
             "standard_steps": 64, "forward_budget": 128,
             "dard": {"temperature": .3, "top_p": .95, "block_size": 128,
                      "steps_per_block": 16, "tau_c": .5, "tau_u": .8}}
    value.update(updates)
    return value


def tasks():
    return [{"dataset": "fixture", "source_id": f"Source/{source}", "language": language,
             "task_id": f"{source}/{language}", "prompt": "Implement correctly"}
            for source in range(2) for language in ("py", "js")]


def phase(tmp_path, name, configs, scores, *, seconds=None, seed=7, split="validation", task_rows=None):
    root = tmp_path/name
    root.mkdir()
    records = tasks() if task_rows is None else task_rows
    manifest = root/"tasks.jsonl"
    manifest.write_text("".join(json.dumps(task)+"\n" for task in records))
    plan = {"split": split, "seed": seed, "tasks_path": str(manifest), "tasks_sha256": file_sha(manifest),
            "candidates": configs, "output": str(root)}
    (root/"evaluation_plan.json").write_text(json.dumps(plan))
    for cfg in configs:
        folder = root/"shard-0/cases"/cfg["id"]
        folder.mkdir(parents=True)
        for index, task in enumerate(records):
            passed = scores[cfg["id"]][index]
            times = (seconds or {}).get(cfg["id"], 1.)
            row = {"split": split, "candidate_id": cfg["id"], "config_sha256": digest(cfg),
                "task_key": task_id(task), "task_sha256": digest(task), "source_id": task["source_id"],
                "language": task["language"], "passed": passed, "status": "passed" if passed else "test_failed",
                "generation_seconds": times[index] if isinstance(times, list) else times,
                "forward_calls": 64 if cfg["decoder"] == "standard" else 10,
                "forced_tokens": int(cfg["decoder"] == "dard"), "infrastructure_error": False,
                "incomplete": False, "verification_certified": True, "unresolved_mask_tokens": 0}
            (folder/(task_id(task)+".json")).write_text(json.dumps(row))
    # The analyzer must recompute from raw receipts instead of trusting this.
    (root/"results.json").write_text('{"candidates":[{"passed":999999}]}')
    return root


def edit_case(root, cid, index, **fields):
    path = root/"shard-0/cases"/cid/(task_id(tasks()[index])+".json")
    row = json.loads(path.read_text())
    row.update(fields)
    path.write_text(json.dumps(row))
    return path


def run(paths, **kwargs):
    return analysis.analyze_evaluations(paths, bootstrap_repetitions=1000, **kwargs)


def summary(report, cid):
    return next(row for row in report["summaries"] if row["candidate_id"] == cid)


def test_holdout_is_rejected_before_any_selection(tmp_path):
    p = phase(tmp_path, "holdout", [config("s")], {"s": [True]*4}, split="holdout")
    with pytest.raises(ValueError, match="holdout is forbidden"):
        run([p])


@pytest.mark.parametrize("change", [
    {"model": "other"}, {"revision": "other"}, {"prefix": "none"}, {"tokens": 256},
    {"adapter": {"path": "/irrelevant", "sha256": {"adapter_config.json": "a"*64, "adapter_model.safetensors": "b"*64}}},
    {"dard": {"temperature": 0., "top_p": .95}}, {"dard": {"temperature": .3, "top_p": 1.}},
])
def test_comparable_cohort_keeps_every_required_generation_condition(change):
    original = config("a")
    changed = deepcopy(original); changed.update(change)
    assert analysis.comparable_settings(original, "tasks", [7]) != analysis.comparable_settings(changed, "tasks", [7])
    assert analysis.comparable_settings(original, "tasks", [7]) != analysis.comparable_settings(original, "other-tasks", [7])
    assert analysis.comparable_settings(original, "tasks", [7]) != analysis.comparable_settings(original, "tasks", [8])


def test_adapter_hashes_define_identity_instead_of_adapter_path():
    adapter = {"path": "/old/path", "sha256": {"adapter_config.json": "a"*64, "adapter_model.safetensors": "b"*64}}
    a = config("a", adapter=adapter)
    b = deepcopy(a); b["adapter"]["path"] = "/new/path"
    assert analysis.comparable_settings(a, "tasks", [7]) == analysis.comparable_settings(b, "tasks", [7])
    b["adapter"]["sha256"] = {}
    with pytest.raises(ValueError, match="SHA256"):
        analysis.comparable_settings(b, "tasks", [7])


def test_correctness_is_independent_of_completion_and_latency_includes_wrong_cases(tmp_path):
    p = phase(tmp_path, "validation", [config("d", decoder="dard")],
              {"d": [True, True, False, False]}, seconds={"d": [1.,1.,9.,9.]})
    edit_case(p, "d", 0, incomplete=True, verification_certified=False, unresolved_mask_tokens=2)
    result = summary(run([p]), "d")
    assert result["valid_for_selection"]
    assert result["passed"] == 2 and result["pass_rate"] == .5
    assert result["completion_qualified_passed"] == 1
    assert result["incomplete_output_count"] == 1
    assert result["unresolved_mask_tokens"] == 2
    assert result["mean_generation_seconds"] == 5
    assert result["p95_generation_seconds"] == 9
    assert result["mean_actual_forward_calls"] == 10
    assert result["forced_case_rate"] == 1


def test_infrastructure_failure_disqualifies_candidate_without_silently_dropping_case(tmp_path):
    p = phase(tmp_path, "validation", [config("bad"), config("good", decoder="dard")],
              {"bad": [True]*4, "good": [True, True, False, False]})
    edit_case(p, "bad", 0, infrastructure_error=True)
    report = run([p])
    bad = summary(report, "bad")
    assert not bad["valid_for_selection"]
    assert bad["observed_cases"] == 4 and bad["infrastructure_error_count"] == 1
    assert report["cohorts"][0]["quality_maximum"]["candidate_id"] == "good"


def test_missing_and_tampered_cases_are_visible_and_cannot_win(tmp_path):
    p = phase(tmp_path, "validation", [config("s")], {"s": [True]*4})
    path = edit_case(p, "s", 0, task_sha256="tampered")
    (path.parent/(task_id(tasks()[1])+".json")).unlink()
    report = run([p]); row = summary(report, "s")
    assert row["observed_cases"] == 2 and row["missing_or_invalid_cases"] == 2
    assert row["pass_rate"] == .5 and not row["valid_for_selection"]
    assert report["cohorts"][0]["quality_maximum"] is None


def test_task_manifest_hash_is_checked(tmp_path):
    p = phase(tmp_path, "validation", [config("s")], {"s": [True]*4})
    (p/"tasks.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="task file changed"):
        run([p])


def test_repeated_seeds_aggregate_but_duplicate_seed_phase_is_invalid(tmp_path):
    a = phase(tmp_path, "a", [config("s")], {"s": [True]*4}, seed=7)
    b = phase(tmp_path, "b", [config("s")], {"s": [False]*4}, seed=8)
    report = run([a,b]); row = summary(report, "s")
    assert row["expected_cases"] == 8 and row["passed"] == 4 and row["source_clusters"] == 2
    assert row["valid_for_selection"] and row["seeds"] == [7,8]
    duplicate = phase(tmp_path, "duplicate", [config("s")], {"s": [True]*4}, seed=7)
    row = summary(run([a,duplicate]), "s")
    assert not row["valid_for_selection"]
    assert any(issue["type"] == "duplicate_candidate_seed_phase" for issue in row["issues"])


def test_shortlist_filter_omits_nonrepeated_candidates_and_unknown_ids_fail(tmp_path):
    a = phase(tmp_path, "full", [config("s"),config("excluded", decoder="dard")],
              {"s": [True]*4, "excluded": [True]*4}, seed=7)
    b = phase(tmp_path, "repeat", [config("s")], {"s": [True]*4}, seed=8)
    report = run([a,b], candidate_ids=["s"])
    assert len(report["summaries"]) == 1 and report["summaries"][0]["seeds"] == [7,8]
    assert report["candidate_id_filter"] == ["s"]
    with pytest.raises(ValueError, match="absent"):
        run([a,b], candidate_ids=["missing"])


def test_bootstrap_groups_languages_and_repeated_seeds_by_source(tmp_path):
    configs = [config("s"), config("d", decoder="dard")]
    scores = {"s": [False,False,True,True], "d": [True,True,False,False]}
    a = phase(tmp_path, "a", configs, scores, seed=7)
    b = phase(tmp_path, "b", configs, scores, seed=8)
    records,_ = analysis.load_evaluations([a,b])
    left = next(list(row["rows"].values()) for row in records if "s" in row["candidate_ids"])
    right = next(list(row["rows"].values()) for row in records if "d" in row["candidate_ids"])
    paired = analysis.paired_source_bootstrap(left, right, repetitions=1000, seed=123)
    reversed_pair = analysis.paired_source_bootstrap(list(reversed(left)), list(reversed(right)), repetitions=1000, seed=123)
    assert paired == reversed_pair
    assert paired["cases"] == 8 and paired["source_clusters"] == 2
    assert paired["candidate_only_correct"] == paired["reference_only_correct"] == 4
    assert paired["accuracy_difference_ci"] == [-1.,1.]
    with pytest.raises(ValueError, match="identical"):
        analysis.paired_source_bootstrap(left[:-1], right, repetitions=100)


def test_pareto_and_practical_choice_keep_quality_uncertainty_and_latency_budgets(tmp_path):
    configs = [config("best"),config("fast", decoder="dard"),config("dominated", decoder="dard", standard_steps=32)]
    p = phase(tmp_path, "validation", configs,
              {"best": [True]*4, "fast": [True,True,True,False], "dominated": [True,False,False,False]},
              seconds={"best": 10., "fast": 1., "dominated": 20.})
    report = run([p], quality_tolerance=.03, latency_budgets=[.5,1.,10.])
    c = report["cohorts"][0]
    assert c["quality_maximum"]["candidate_id"] == c["practical_choice"]["candidate_id"] == "best"
    assert len(c["pareto_frontier"]) == 2
    assert [row["candidate_id"] for row in c["latency_budget_operating_points"]] == [None,"fast","best"]
    assert not c["practical_choice"]["statistical_noninferiority_established"]
    permissive = run([p], quality_tolerance=1.)
    assert permissive["cohorts"][0]["practical_choice"]["candidate_id"] == "fast"


def test_ar_gate_is_explicit_cross_model_target_and_keeps_frontiers_separate(tmp_path):
    p = phase(tmp_path, "validation", [config("ar",decoder="ar",model="reference/ar"), config("d",decoder="dard")],
              {"ar": [True]*4, "d": [True]*4}, seconds={"ar": 2., "d": 1.})
    report = run([p])
    assert len(report["cohorts"]) == 2
    gate = report["ar_accuracy_gates"][0]
    assert gate["empirical_accuracy_target_met"] and gate["exploratory_paired_ci_target_met"]
    assert gate["candidate_faster_than_ar"]
    assert not gate["statistical_noninferiority_established"]
    assert not report["statistical_policy"]["multiplicity_adjusted"]


def test_exports_are_standalone_and_do_not_modify_evaluation_receipts(tmp_path):
    p = phase(tmp_path, "validation", [config("s", model="model/<name>&value")], {"s": [True]*4})
    before = {str(path.relative_to(p)):path.read_bytes() for path in p.rglob("*") if path.is_file()}
    output = tmp_path/"analysis"
    report = run([p], output=output)
    saved = json.loads((output/"tradeoff_analysis.json").read_text())
    assert saved == report
    with (output/"tradeoff_summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["passed"] == "4"
    ET.parse(output/"tradeoff.svg")
    assert before == {str(path.relative_to(p)):path.read_bytes() for path in p.rglob("*") if path.is_file()}


def test_generation_receipt_identity_and_digest_are_verified(tmp_path):
    p = phase(tmp_path, "validation", [config("s")], {"s": [True]*4})
    path = edit_case(p, "s", 0)
    row = json.loads(path.read_text()); receipt = p/"generation.json"
    receipt.write_text(json.dumps({key: row[key] for key in ("candidate_id","task_key","task_sha256","config_sha256")}))
    edit_case(p, "s", 0, generation_receipt=str(receipt), generation_receipt_sha256=file_sha(receipt))
    assert summary(run([p]), "s")["valid_for_selection"]
    receipt.write_text("{}")
    assert not summary(run([p]), "s")["valid_for_selection"]


def test_macro_language_sensitivity_does_not_replace_micro_selection(tmp_path):
    unequal = [*tasks(), {"dataset":"fixture", "source_id":"Source/2", "language":"py",
                         "task_id":"2/py", "prompt":"Implement correctly"}]
    p = phase(tmp_path, "validation", [config("s")], {"s":[True,False,True,False,True]}, task_rows=unequal)
    report = run([p]); row = summary(report, "s")
    assert row["pass_rate"] == .6 and row["macro_language_pass_rate"] == .5
    assert row["by_language"]["py"]["pass_rate"] == 1.
    assert row["by_language"]["js"]["pass_rate"] == 0.
    assert report["primary_selection_metric"] == "micro_functional_pass_rate"


def test_recorded_measurement_seed_must_match_plan(tmp_path):
    p = phase(tmp_path, "validation", [config("s")], {"s":[True]*4})
    edit_case(p, "s", 0, measurement_seed=999)
    assert not summary(run([p]), "s")["valid_for_selection"]


def test_different_extraction_or_timing_protocols_are_never_pooled(tmp_path):
    a = phase(tmp_path, "legacy", [config("s")], {"s":[True]*4}, seed=7)
    b = phase(tmp_path, "new", [config("s")], {"s":[True]*4}, seed=8)
    plan = json.loads((b/"evaluation_plan.json").read_text()); plan["extraction_version"] = "v2"
    (b/"evaluation_plan.json").write_text(json.dumps(plan))
    report = run([a,b])
    assert len(report["cohorts"]) == len(report["summaries"]) == 2
    assert all(row["expected_cases"] == 4 for row in report["summaries"])
