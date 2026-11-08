"""Campaign aggregation uses paired totals and retains failure/retry costs."""
import json
from pathlib import Path
import tempfile
import unittest

from campaign_reporting import summarize_campaign, summarize_campaign_collection, write_priority_index, write_goal_report
from experiment_analysis import analyze_run
from test_experiment_analysis import function_case, function_config


class CampaignReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.plan = {"kind": "functions", "repeats": 1, "task_count": 4, "expected_cases": 12,
                     "workers": [{"task_keys": [["humaneval", "py", f"task-{index}"]]} for index in range(4)]}

    def report(self, *, unknown=False, contended=False, retry=False, missing=False):
        shards = []
        for index, (baseline, candidate) in enumerate(((1, 1), (9, 3), (1, 1), (9, 3))):
            run = self.root / f"shard-{index}"
            run.mkdir(exist_ok=True)
            config = function_config()
            rows = []
            for decoder in self.plan.get("backends", ("ar", "standard", "dard")):
                elapsed = candidate if decoder == "dard" else baseline
                correct = not (decoder == "dard" and index == 3)
                row = function_case(decoder, task_id=f"task-{index}", generation_wall_s=elapsed,
                    first_generation_wall_s=elapsed, agent_wall_s=elapsed + 1, total_wall_s=elapsed + 2,
                    offline_grading_wall_s=1, first_pass=correct, final_pass=correct,
                    final_status="passed" if correct else "test_failed")
                if index == 0 and decoder == "dard":
                    if unknown:
                        row.update(case_completed=False, final_pass=None, first_pass=None,
                                   final_status="infra_error", failure_kind="grading_infrastructure")
                    if contended:
                        row["resource_contended"] = True
                if not (missing and index == 0 and decoder == "ar"):
                    rows.append(row)
                phase = config["phases"][{"ar": 0, "standard": 1, "dard": 2}[decoder]]
                phase.update(decoder=decoder, phase_wall_s=elapsed + 2, model_load_s=1, warmup_s=1)
            if retry and index == 0:
                rows.insert(0, function_case("dard", task_id=f"task-{index}", case_completed=False,
                    first_pass=None, final_pass=None, total_wall_s=7, failure_kind="generation_error"))
                config["phases"].append({"phase_id": 99, "decoder": "dard", "phase_wall_s": 7})
            (run / "config.json").write_text(json.dumps(config))
            (run / "case_results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            shards.append({"rows": rows, "diagnostics": analyze_run(run), "phases": config["phases"]})
        return summarize_campaign(self.plan, shards, {"coverage_complete": not missing}, self.root)

    def test_all_three_comparisons_use_sum_ratio_and_include_failed_solutions(self):
        report = self.report()
        pair = report["paired_comparisons"]["dard_vs_ar/all"]
        self.assertTrue(pair["eligible_for_controlled_comparison"])
        self.assertEqual(pair["speedup_ratio_of_summed_times"]["generation_wall_s"], 2.5)
        self.assertEqual(pair["final_pass_rate_delta_dard_minus_baseline"], -.25)
        self.assertEqual(report["groups"]["dard/all"]["failed_cases"], 1)
        self.assertEqual(report["groups"]["dard/all"]["summed_latest_case_seconds"]["generation_wall_s"], 8)
        standard = report["paired_comparisons"]["standard_vs_ar/all"]
        self.assertEqual(standard["speedup_ratio_of_summed_times"]["generation_wall_s"], 1)
        self.assertIsNone(pair["pooled_confidence_interval"])

    def test_infrastructure_unknowns_and_missing_cases_cannot_certify_speed(self):
        for options in ({"unknown": True}, {"missing": True}):
            report = self.report(**options)
            pair = report["paired_comparisons"]["dard_vs_ar/all"]
            self.assertFalse(pair["eligible_for_controlled_comparison"])
            self.assertIsNone(pair["speedup_ratio_of_summed_times"]["generation_wall_s"])
            self.assertIsNone(report["groups"]["dard/all"]["full_cohort_final_pass_rate"])

    def test_dard_contention_does_not_hide_clean_standard_ar_comparison(self):
        report = self.report(contended=True)
        self.assertFalse(report["paired_comparisons"]["dard_vs_ar/all"]["eligible_for_controlled_comparison"])
        self.assertTrue(report["paired_comparisons"]["standard_vs_ar/all"]["eligible_for_controlled_comparison"])

    def test_retry_work_and_phase_costs_are_reported_without_double_counting_outcomes(self):
        report = self.report(retry=True)
        self.assertEqual(report["recorded_latest_cases"], 12)
        self.assertEqual(report["raw_case_rows"], 13)
        self.assertEqual(report["retry_accounting"]["superseded_recorded_case_wall_s"], 7)
        group = report["groups"]["dard/all"]
        self.assertEqual(group["passed_cases"], 3)
        self.assertEqual(group["phase_accounting"]["summed_phase_wall_s"], 23)
        self.assertLess(group["phase_accounting"]["successful_cases_per_summed_phase_hour"],
                        group["successful_cases_per_summed_task_hour"])

    def test_priority_two_backend_campaign_has_only_planned_groups_and_comparison(self):
        self.plan.update(backends=["ar", "dard"], expected_cases=8)
        report = self.report()
        self.assertEqual(report["recorded_latest_cases"], 8)
        self.assertEqual(report["backends"], ["ar", "dard"])
        self.assertFalse(any(key.startswith("standard/") for key in report["groups"]))
        self.assertEqual(set(report["paired_comparisons"]), {"dard_vs_ar/all", "dard_vs_ar/humaneval/py"})
        self.assertTrue(report["paired_comparisons"]["dard_vs_ar/all"]["eligible_for_controlled_comparison"])

    def test_optional_standard_only_has_no_missing_ar_or_dard_comparison(self):
        self.plan.update(backends=["standard"], expected_cases=4)
        report = self.report()
        self.assertEqual(report["paired_comparisons"], {})
        self.assertEqual(set(report["groups"]), {"standard/all", "standard/humaneval/py"})
        self.assertTrue(report["groups"]["standard/all"]["complete_evaluated_cohort"])

    def test_collection_preserves_priority_completeness_without_optional_run(self):
        self.plan.update(backends=["ar", "dard"], expected_cases=8)
        original = self.report()
        output = self.root / "combined"
        result = summarize_campaign_collection({"priority": self.root, "standard": self.root / "missing"}, output)
        self.assertTrue(result["required_campaigns_complete"])
        self.assertFalse(result["all_available_campaigns_complete"])
        self.assertFalse(result["campaigns"]["standard"]["available"])
        self.assertEqual(result["campaigns"]["priority"]["report"], original)
        self.assertEqual(result["cross_campaign_comparisons"], [])
        self.assertTrue((output / "combined_campaign_summary.json").exists())

    def test_collection_does_not_call_unknown_outcomes_complete(self):
        self.plan.update(backends=["ar", "dard"], expected_cases=8)
        self.report(unknown=True)
        result = summarize_campaign_collection({"priority": self.root}, self.root / "combined")
        self.assertFalse(result["required_campaigns_complete"])

    def test_priority_index_keeps_exact_counts_timing_budget_and_source_links(self):
        self.plan.update(backends=["ar", "dard"], expected_cases=8)
        report = self.report()
        run_root = self.root / "indexed"
        priority = run_root / "functions-priority"
        priority.mkdir(parents=True)
        (priority / "campaign_summary.json").write_text(json.dumps(report))
        decision = {"deadline_unix": 123456, "standard_action": "skip", "remaining_s": 0}
        (run_root / "budget_decision.json").write_text(json.dumps(decision))
        result = write_priority_index(run_root)
        self.assertEqual(result["budget_decision"], decision)
        self.assertEqual(result["summary_groups"]["priority"]["expected_cases"], 8)
        self.assertEqual(result["summary_groups"]["priority"]["groups"]["dard/all"]["passed_cases"], 3)
        self.assertEqual(result["campaigns"]["priority"]["report"], report)
        self.assertTrue(result["required_campaigns_complete"])
        markdown = (run_root / "priority_summary.md").read_text()
        self.assertIn(str(priority / "campaign_summary.json"), markdown)
        self.assertIn("| priority | dard | 4 | 4 | 4 | 3 | 1 |", markdown)
        self.assertIn('"standard_action": "skip"', markdown)
        self.assertEqual(json.loads((run_root / "priority_summary.json").read_text()), result)

    def test_priority_index_can_be_written_before_either_campaign_starts(self):
        result = write_priority_index(self.root / "not-started")
        self.assertFalse(result["required_campaigns_complete"])
        self.assertIsNone(result["budget_decision"])


class GoalReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.plan = {"kind": "functions", "repeats": 1, "task_count": 8, "expected_cases": 24,
                     "backends": ["ar", "dard", "standard"], "workers": [],
                     "selection_manifest": {"schema_version": 1, "task_count": 8}}
        for index, language in enumerate(("py", "js", "rs", "cpp")):
            run = self.root / f"shard-{index}"
            run.mkdir()
            dataset = "humaneval" if language == "py" else "multiple_" + language
            config = function_config()
            config["settings"].update(repairs=0, trial_seeds=[42],
                                      local_environments={language: {"version": f"fixture-{language}"}})
            config["expected_groups"] = {f"{decoder}/{dataset}/{language}": 2 for decoder in self.plan["backends"]}
            rows = []
            for number in range(2):
                for decoder in self.plan["backends"]:
                    elapsed = {"ar": 4, "standard": 3, "dard": 2}[decoder]
                    correct = decoder != "dard" or number == 0
                    rows.append(function_case(decoder, dataset=dataset, language=language,
                                task_id=f"task-{number}", source_id=f"HumanEval/{number}",
                                first_generation_wall_s=elapsed, generation_wall_s=elapsed,
                                total_wall_s=elapsed+2, first_pass=correct, final_pass=correct,
                                final_status="passed" if correct else "test_failed"))
            (run / "config.json").write_text(json.dumps(config))
            (run / "case_results.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
            self.plan["workers"].append({"output": str(run), "task_seeds": [[42], [42]],
                "task_keys": [[dataset, language, f"task-{number}"] for number in range(2)]})
        (self.root / "campaign.json").write_text(json.dumps(self.plan))

    def edit_rows(self, edit, shard=0):
        path = self.root / f"shard-{shard}" / "case_results.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows = edit(rows)
        path.write_text("".join(json.dumps(row)+"\n" for row in rows))

    def report(self):
        return write_goal_report(self.root, bootstrap_draws=100)

    def test_all_languages_three_pairs_cluster_translations_and_separate_latency(self):
        report = self.report()
        self.assertEqual(report["protocol_issues"], [])
        self.assertTrue(report["coverage_complete"])
        self.assertTrue(report["complete_evaluated_campaign"])
        self.assertEqual(set(report["languages"]), {"py", "js", "rs", "cpp"})
        self.assertEqual(report["selected_unique_source_problems"], 2)
        self.assertEqual(report["common_source_problems_across_languages"], 2)
        self.assertEqual(len(report["comparisons"]), 15)
        group = report["groups"]["dard/py"]
        self.assertEqual(group["pass_at_1"], .5)
        self.assertEqual(group["generation_latency"]["sum_s"], 4)
        self.assertEqual(group["generation_latency"]["median_s"], 2)
        self.assertEqual(group["generation_latency"]["p95_s"], 2)
        self.assertEqual(group["end_to_end_latency"]["sum_s"], 8)
        pair = report["comparisons"]["dard_vs_ar/all"]
        self.assertTrue(pair["complete_known_controlled_cohort"])
        self.assertEqual(pair["unique_source_clusters"], 2)  # Not eight independent translations.
        self.assertEqual(pair["pass_at_1_delta_candidate_minus_baseline"], -.5)
        self.assertEqual(pair["pass_at_1_delta_source_cluster_bootstrap_95ci"], [-1, 0])
        self.assertEqual(pair["generation_speedup_baseline_over_candidate"], 2)
        self.assertEqual(pair["end_to_end_speedup_baseline_over_candidate"], 1.5)
        self.assertIsNone(pair["quality_similarity_established"])
        self.assertIsNone(report["overall_goal_established"])
        self.assertEqual(json.loads((self.root / "goal_report.json").read_text()), report)
        self.assertIn("Python", (self.root / "goal_report.md").read_text())

    def test_missing_and_unknown_pairs_never_publish_quality_or_speed_claims(self):
        self.edit_rows(lambda rows: [row for row in rows if not (row["decoder"] == "dard" and row["task_id"] == "task-0")])
        report = self.report()
        pair = report["comparisons"]["dard_vs_ar/all"]
        self.assertFalse(pair["complete_known_controlled_cohort"])
        self.assertFalse(report["coverage_complete"])
        self.assertFalse(report["complete_evaluated_campaign"])
        self.assertIsNone(pair["pass_at_1_delta_candidate_minus_baseline"])
        self.assertIsNone(pair["generation_speedup_baseline_over_candidate"])
        self.assertTrue(report["comparisons"]["standard_vs_ar/all"]["complete_known_controlled_cohort"])
        self.assertTrue(report["comparisons"]["dard_vs_ar/js"]["complete_known_controlled_cohort"])

    def test_unknown_grade_is_not_counted_as_an_incorrect_answer(self):
        def edit(rows):
            rows[1].update(first_pass=None, final_pass=None, case_completed=False)
            return rows
        self.edit_rows(edit)
        report = self.report()
        self.assertTrue(report["coverage_complete"])
        self.assertFalse(report["complete_evaluated_campaign"])
        group = report["groups"]["dard/py"]
        self.assertEqual(group["evaluated_cases"], 1)
        self.assertIsNone(group["pass_at_1"])
        pair = report["comparisons"]["dard_vs_ar/py"]
        self.assertIsNone(pair["pass_at_1_delta_source_cluster_bootstrap_95ci"])
        self.assertIsNone(pair["generation_speedup_baseline_over_candidate"])

    def test_repair_protocol_or_mismatched_shards_prevent_single_pass_report_claims(self):
        config_path = self.root / "shard-0" / "config.json"
        config = json.loads(config_path.read_text())
        config["settings"]["repairs"] = 1
        config_path.write_text(json.dumps(config))
        report = self.report()
        self.assertIn("requires_zero_repairs", report["protocol_issues"])
        self.assertIn("different_shard_protocol_or_environment", report["protocol_issues"])
        self.assertIsNone(report["comparisons"]["dard_vs_ar/all"]["generation_speedup_baseline_over_candidate"])

    def test_inconsistent_single_submission_and_unplanned_identity_block_comparison(self):
        def edit(rows):
            rows[0].update(attempts=2)
            rows[-1]["task_id"] = "unplanned"
            return rows
        self.edit_rows(edit)
        report = self.report()
        self.assertIn("inconsistent_single_submission_record", report["protocol_issues"])
        self.assertIn("duplicate_or_unplanned_recorded_cases", report["protocol_issues"])
        self.assertFalse(report["comparisons"]["standard_vs_ar/all"]["complete_known_controlled_cohort"])

    def test_source_cluster_ci_is_reproducible_and_does_not_modify_raw_evidence(self):
        before = {str(path): path.read_bytes() for path in self.root.glob("shard-*/*.json*")}
        first, second = self.report(), self.report()
        self.assertEqual(first, second)
        self.assertEqual(before, {str(path): path.read_bytes() for path in self.root.glob("shard-*/*.json*")})

    def test_different_language_environment_scopes_are_valid_but_shared_drift_is_not(self):
        # The fixture already has a different singleton runtime on each shard.
        clean = self.report()
        self.assertTrue(clean["complete_evaluated_campaign"])
        self.assertTrue(clean["comparisons"]["dard_vs_ar/all"]["complete_known_controlled_cohort"])
        path = self.root / "shard-1" / "config.json"
        config = json.loads(path.read_text())
        config["settings"]["local_environments"]["py"] = {"version": "fixture-py"}
        path.write_text(json.dumps(config))
        self.assertEqual(self.report()["protocol_issues"], [])
        config["settings"]["local_environments"]["py"]["version"] = "different-py"
        path.write_text(json.dumps(config))
        report = self.report()
        self.assertIn("different_shard_protocol_or_environment", report["protocol_issues"])
        self.assertFalse(report["complete_evaluated_campaign"])

    def test_missing_required_language_environment_is_not_silently_accepted(self):
        path = self.root / "shard-0" / "config.json"
        config = json.loads(path.read_text())
        config["settings"]["local_environments"] = {}
        path.write_text(json.dumps(config))
        report = self.report()
        self.assertIn("missing_required_local_environment_identity", report["protocol_issues"])
        self.assertFalse(report["complete_evaluated_campaign"])

    def test_official_suite_requires_its_evaluator_environment_identity(self):
        path = self.root / "shard-0" / "config.json"
        config = json.loads(path.read_text())
        config["expected_groups"] = {key.replace("/humaneval/", "/humaneval_plus/"): value
                                      for key, value in config["expected_groups"].items()}
        path.write_text(json.dumps(config))
        self.edit_rows(lambda rows: [dict(row, dataset="humaneval_plus") for row in rows])
        for task in self.plan["workers"][0]["task_keys"]:
            task[0] = "humaneval_plus"
        (self.root / "campaign.json").write_text(json.dumps(self.plan))
        self.assertIn("missing_required_local_environment_identity", self.report()["protocol_issues"])
        config["settings"]["local_environments"]["official:evalplus"] = {"evaluator_version": "0.3.1"}
        path.write_text(json.dumps(config))
        report = self.report()
        self.assertEqual(report["protocol_issues"], [])
        self.assertTrue(report["complete_evaluated_campaign"])


if __name__ == "__main__":
    unittest.main()
