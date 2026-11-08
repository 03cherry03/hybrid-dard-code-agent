"""Offline diagnostics preserve regressions, unknowns and experiment provenance."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiment_analysis import analyze_run, main, write_report
from experiment_journal import append_entry, read_journal


def identity(name="diffusion"):
    return {"model_name": name, "resolved_revision": "fixed-model", "tokenizer_revision": "fixed-tokenizer",
        "tokenizer_vocab_sha256": "vocab", "model_config_sha256": "config", "chat_template_sha256": "template",
        "gpu_uuid": "GPU-1", "cuda_build": "11.8", "versions": {"torch": "2.7.1"},
        "compute_dtype": "bfloat16", "quant": "bf16", "identity_verified": True}


def function_config(count=1):
    return {"settings": {"task_manifest_sha256": "tasks", "tokens": 256, "repairs": 1,
        "temperature": 0, "top_p": 1, "environment": {"python": "fixture"},
        "source_sha256": {"models.py": "source"}},
        "expected_groups": {f"{d}/humaneval/py": count for d in ("ar", "standard", "dard")},
        "phases": [{"phase_id": i, "backend": identity("ar" if d == "ar" else "diffusion")}
            for i, d in enumerate(("ar", "standard", "dard"))]}


def function_case(decoder, **overrides):
    result = {"decoder": decoder, "dataset": "humaneval", "language": "py", "task_id": "task-1",
        "source_id": "task-1", "repeat": 0, "seed": 42, "task_sha256": "task", "messages_sha256": "prompt",
        "phase_id": {"ar": 0, "standard": 1, "dard": 2}[decoder], "model_identity_verified": True,
        "warmup_done": True, "resource_check_available": True, "resource_contended": False,
        "case_completed": True, "first_pass": True, "final_pass": True, "final_status": "passed",
        "generation_wall_s": 4, "model_wall_s": 3, "agent_wall_s": 5, "total_wall_s": 6,
        "model_forward_passes": 20, "forced_tokens": 0, "revoked_tokens": 0, "attention_fallback_steps": 0,
        "output_slots": 32, "attempts": 1, "measurement_complete": True, "diagnostics_mode": "summary"}
    result.update(overrides)
    return result


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def json(self, name, value):
        (self.run / name).write_text(json.dumps(value), encoding="utf-8")

    def jsonl(self, name, rows):
        (self.run / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def function_run(self, dard=None, baseline=None, config=None):
        self.json("config.json", config or function_config())
        self.jsonl("case_results.jsonl", [function_case("ar"), baseline or function_case("standard"),
                                          dard or function_case("dard")])
        return analyze_run(self.run)

    def pair(self, report):
        return next(p for p in report["pairs"] if p["comparison"] == "dard_vs_standard")

    def test_fewer_forwards_can_be_slower_without_claiming_cause(self):
        report = self.function_run(function_case("dard", model_forward_passes=10, generation_wall_s=8))
        pair = self.pair(report)
        self.assertTrue(pair["eligible_for_controlled_comparison"])
        self.assertEqual(pair["forward_difference_dard_minus_baseline"], -10)
        self.assertEqual(pair["observed_generation_speedup_baseline_over_dard"], .5)
        findings = report["groups"]["dard_vs_standard/humaneval/py"]["findings"]
        finding = next(f for f in findings if f["code"] == "fewer_forwards_but_slower")
        self.assertFalse(finding["causal_claim"])
        self.assertIn("case_results.jsonl:3", finding["evidence"])

    def test_quality_regression_and_failure_cost_stay_in_cohort(self):
        report = self.function_run(function_case("dard", first_pass=False, final_pass=False, total_wall_s=12))
        group = report["groups"]["dard_vs_standard/humaneval/py"]
        self.assertEqual(group["final_outcomes"], {"regression": 1})
        self.assertEqual(group["final_pass_rate_delta_dard_minus_baseline"], -1)
        self.assertEqual(group["end_to_end_speedup_baseline_over_dard"], .5)

    def test_forcing_remains_algorithm_behavior_but_fallback_is_ineligible(self):
        report = self.function_run(function_case("dard", forced_tokens=5))
        self.assertTrue(self.pair(report)["eligible_for_controlled_comparison"])
        self.assertTrue(any(f["code"] == "forced_completion_observed" for f in
            report["groups"]["dard_vs_standard/humaneval/py"]["findings"]))
        report = self.function_run(function_case("dard", attention_fallback_steps=1))
        pair = self.pair(report)
        self.assertFalse(pair["eligible_for_controlled_comparison"])
        self.assertIn("diagnostic_attention_fallback", pair["control_issues"])
        self.assertIsNone(pair["observed_generation_speedup_baseline_over_dard"])

    def test_different_seed_is_unmatched_not_paired(self):
        report = self.function_run(function_case("dard", seed=43))
        self.assertEqual([pair["comparison"] for pair in report["pairs"]], ["standard_vs_ar"])
        self.assertEqual(len(report["unmatched"]), 4)
        self.assertFalse(report["groups"]["dard_vs_standard/humaneval/py"]["complete_cohort"])

    def test_different_prompt_or_checkpoint_blocks_interpretable_pair(self):
        report = self.function_run(function_case("dard", messages_sha256="other"))
        pair = self.pair(report)
        self.assertEqual(pair["final_outcome"], "uncontrolled")
        self.assertIsNone(pair["observed_generation_speedup_baseline_over_dard"])
        config = function_config()
        config["phases"][2]["backend"]["resolved_revision"] = "different"
        report = self.function_run(config=config)
        self.assertIn("different_diffusion_checkpoint_or_tokenizer", self.pair(report)["control_issues"])

    def test_profile_and_contended_gpu_preserve_quality_but_not_clean_speed(self):
        for overrides, issue in (({"diagnostics_mode": "profile"}, "instrumentation_affects_timing"),
                                 ({"resource_contended": True}, "other_compute_process_on_selected_gpu"),
                                 ({"measurement_complete": False}, "incomplete_measurement")):
            with self.subTest(issue=issue):
                report = self.function_run(function_case("dard", **overrides))
                pair = self.pair(report)
                self.assertIn(issue, pair["timing_issues"])
                self.assertEqual(pair["final_outcome"], "both_pass")
                self.assertIsNone(pair["observed_generation_speedup_baseline_over_dard"])
                if issue == "instrumentation_affects_timing":
                    self.assertIn("instrumentation_can_change_outcomes_under_time_limits", pair["quality_interpretation_issues"])

    def test_unknown_grades_and_metrics_do_not_become_failure_or_zero(self):
        case = function_case("dard", first_pass=None, final_pass=None)
        del case["forced_tokens"]
        report = self.function_run(case)
        pair = self.pair(report)
        self.assertEqual(pair["final_outcome"], "unknown")
        self.assertIsNone(pair["candidate"]["metrics"]["forced_tokens"])
        self.assertIsNone(report["groups"]["dard_vs_standard/humaneval/py"]["final_pass_rate_delta_dard_minus_baseline"])
        self.assertFalse(pair["eligible_for_controlled_comparison"])
        self.assertIn("unknown_final_outcomes", pair["timing_issues"])
        self.assertIsNone(pair["observed_generation_speedup_baseline_over_candidate"])

    def test_complete_but_ungraded_cohort_blocks_all_three_speed_comparisons(self):
        self.json("config.json", function_config())
        self.jsonl("case_results.jsonl", [function_case(decoder, first_pass=None, final_pass=None,
                   case_completed=True) for decoder in ("ar", "standard", "dard")])
        report = analyze_run(self.run)
        self.assertEqual(len(report["pairs"]), 3)
        for pair in report["pairs"]:
            self.assertEqual(pair["final_outcome"], "unknown")
            self.assertFalse(pair["eligible_for_controlled_comparison"])
            self.assertIsNone(pair["observed_generation_speedup_baseline_over_candidate"])
        for group in report["groups"].values():
            self.assertTrue(group["complete_cohort"])
            self.assertIn("unknown_final_outcomes", group["issues"])
            self.assertFalse(group["eligible_for_controlled_comparison"])
            self.assertIsNone(group["generation_speedup_baseline_over_candidate"])

    def test_known_final_grade_does_not_require_first_attempt_grade(self):
        self.json("config.json", function_config())
        self.jsonl("case_results.jsonl", [function_case(decoder, first_pass=None, final_pass=True)
                   for decoder in ("ar", "standard", "dard")])
        report = analyze_run(self.run)
        self.assertTrue(all(pair["eligible_for_controlled_comparison"] for pair in report["pairs"]))
        self.assertTrue(all(pair["first_outcome"] == "unknown" for pair in report["pairs"]))

    def test_incomplete_cohort_does_not_use_good_subset_to_claim_speed(self):
        report = self.function_run(config=function_config(count=2))
        group = report["groups"]["dard_vs_standard/humaneval/py"]
        self.assertFalse(group["eligible_for_controlled_comparison"])
        self.assertIsNone(group["generation_speedup_baseline_over_dard"])
        self.assertIn("incomplete_paired_evaluation", group["issues"])

    def test_empty_directory_and_prepared_manifest_are_not_experiment_results(self):
        report = write_report(self.run)
        self.assertEqual(report["status"], "no_evaluation_data")
        self.assertEqual(report["cases"], [])
        self.assertIn("unknown", (self.run / "diagnostics.md").read_text())
        self.json("config.json", function_config())
        report = analyze_run(self.run)
        self.assertEqual(report["status"], "no_evaluation_data")
        self.assertTrue(all(g["final_pass_rate_delta_candidate_minus_baseline"] is None for g in report["groups"].values()))

    def test_partial_jsonl_is_not_silently_dropped(self):
        self.json("config.json", function_config())
        (self.run / "case_results.jsonl").write_text('{"decoder":')
        with self.assertRaisesRegex(ValueError, "case_results.jsonl:1"):
            analyze_run(self.run)

    def test_generation_diagnostics_aggregate_with_raw_evidence_only(self):
        self.function_run()
        row = function_case("dard")
        row.update(attempt=1, code="DO_NOT_COPY_CODE", generation={
            "wall_s": 4, "budget_exhausted": True, "unfilled_mask_tokens": 2,
            "diagnostics": {"mode": "profile", "complete": True, "stage_wall_s": {"tokenize": .1},
                "forward_shapes": {"sum_input_tokens": 400},
                "decoder": {"component_wall_s": {"verification": 2}}}})
        self.jsonl("runs.jsonl", [row])
        report = analyze_run(self.run)
        metrics = self.pair(report)["candidate"]["metrics"]
        self.assertEqual(metrics["decoder_phase_wall_s"]["verification"]["sum"], 2)
        self.assertEqual(metrics["budget_exhausted_calls"], 1)
        self.assertEqual(metrics["unfilled_mask_tokens"], 2)
        self.assertNotIn("DO_NOT_COPY_CODE", json.dumps(report))

    def test_gpu_peak_falls_back_to_max_observed_partial_or_completed_calls(self):
        self.function_run()
        self.jsonl("runs.jsonl", [dict(function_case("dard"), attempt=1, generation={"gpu_peak_gib": 2.5}),
                                  dict(function_case("dard"), attempt=2, generation={"gpu_peak_gib": 3.5})])
        report = analyze_run(self.run)
        candidate = self.pair(report)["candidate"]["metrics"]
        self.assertEqual(candidate["max_gpu_peak_gib"], 3.5)
        self.assertEqual(candidate["gpu_peak_calls_observed"], 2)
        self.assertIsNone(self.pair(report)["baseline"]["metrics"]["max_gpu_peak_gib"])

    def test_hard_worker_unavailable_metrics_are_unknown_not_legacy_zero(self):
        self.function_run(function_case("dard", measurement_complete=False, model_forward_passes=0,
            model_wall_s=0, max_gpu_peak_gib=0, generation_wall_s=12, agent_wall_s=13, total_wall_s=14))
        self.jsonl("runs.jsonl", [dict(function_case("dard"), attempt=1,
            generation={"measurement_complete": False,
                "diagnostics": {"complete": False, "status": "unavailable_worker_terminated"}})])
        metrics = self.pair(analyze_run(self.run))["candidate"]["metrics"]
        for name in ("model_forward_passes", "model_wall_s", "max_gpu_peak_gib", "forced_tokens"):
            self.assertIsNone(metrics[name], name)
        self.assertEqual(metrics["generation_wall_s"], 12)
        self.assertEqual(metrics["agent_wall_s"], 13)
        self.assertEqual(metrics["total_wall_s"], 14)

    def test_known_partial_metrics_survive_and_mixed_missing_calls_are_explicit(self):
        self.function_run(function_case("dard", measurement_complete=False, model_forward_passes=0,
                                       model_wall_s=0, max_gpu_peak_gib=0))
        partial = {"measurement_complete": False, "model_forward_passes": 7, "model_wall_s": 2.5,
                   "gpu_peak_gib": 3.5, "diagnostics": {"complete": False, "phase": "model_forward"}}
        self.jsonl("runs.jsonl", [dict(function_case("dard"), attempt=1, generation=partial)])
        metrics = self.pair(analyze_run(self.run))["candidate"]["metrics"]
        self.assertEqual(metrics["model_forward_passes"], 7)
        self.assertEqual(metrics["model_wall_s"], 2.5)
        self.assertEqual(metrics["max_gpu_peak_gib"], 3.5)
        self.jsonl("runs.jsonl", [dict(function_case("dard"), attempt=1, generation=partial),
            dict(function_case("dard"), attempt=2, generation={"measurement_complete": False,
                "diagnostics": {"complete": False, "status": "unavailable_worker_terminated"}})])
        metrics = self.pair(analyze_run(self.run))["candidate"]["metrics"]
        self.assertIsNone(metrics["model_forward_passes"])
        self.assertIsNone(metrics["model_wall_s"])
        self.assertIsNone(metrics["max_gpu_peak_gib"])
        coverage = metrics["partial_generation_measurements"]
        self.assertEqual(coverage["model_forward_passes"]["observed_sum"], 7)
        self.assertEqual(coverage["model_forward_passes"]["calls_observed"], 1)
        self.assertEqual(coverage["model_forward_passes"]["calls_recorded"], 2)
        self.assertEqual(coverage["max_gpu_peak_gib"]["observed_max"], 3.5)

    def test_repository_invalid_actions_and_missing_grades_are_visible(self):
        config = {"task_set_sha256": "tasks", "tasks": [{"dataset": "swe", "language": "py"}],
            "decoders": ["standard", "dard"], "repeats": 1, "warmup": 1,
            "agent_config": {"max_steps": 30, "max_context_chars": 30000, "max_input_tokens": 8192,
                "max_output_tokens": 1024, "max_tool_calls": 30, "task_timeout_s": 600,
                "tool_timeout_s": 90, "max_tool_output_chars": 10000}, "temperature": 0, "top_p": 1,
            "runtime": {"python": "fixture"}, "source_sha256": {"models.py": "source"}}
        self.json("run.json", config)
        rows = []
        traces = []
        for decoder in ("standard", "dard"):
            rows.append({"dataset": "swe", "language": "py", "instance_id": "repo-1", "decoder": decoder,
                "seed": 4, "repeat": 0, "task_sha256": "task", "initial_messages_sha256": "prompt",
                "model_metadata": identity(), "environment_id": "sha256:image", "status": "submitted",
                "generation_wall_s": 4, "agent_wall_s": 5, "total_wall_s": 6, "steps": 2,
                "gpu_before": {"verified": True, "other_pids": []}, "gpu_after": {"verified": True, "other_pids": []}})
            if decoder == "dard":
                rows[-1].update(tool_error_count=1, context_trim_events=1)
            traces.append({"key": ["swe", "repo-1", decoder, 0], "trace": [{"step": 1,
                "response": "SECRET_GENERATED_OUTPUT", "generation": {"model_forward_passes": 4},
                "tool_output": {"source_truncated": decoder == "dard", "context_truncated": False},
                "observation": {"ok": False, "error": "Invalid action: invalid JSON"} if decoder == "dard" else
                               {"ok": True}}]})
        self.jsonl("results.jsonl", rows)
        self.jsonl("traces.jsonl", traces)
        report = analyze_run(self.run)
        pair = self.pair(report)
        self.assertEqual(pair["candidate"]["metrics"]["invalid_actions"], 1)
        self.assertEqual(pair["final_outcome"], "unknown")
        self.assertNotIn("SECRET_GENERATED_OUTPUT", json.dumps(report))
        codes = {f["code"] for f in report["groups"]["dard_vs_standard/swe/py"]["findings"]}
        self.assertTrue({"invalid_repository_actions", "repository_context_trimmed", "repository_tool_errors",
                         "repository_tool_output_truncated"} <= codes)

    def test_no_cases_can_still_expose_sanitized_error_stage_evidence(self):
        config = function_config()
        config["phases"][0]["error"] = "RuntimeError: DO_NOT_COPY_ERROR"
        self.json("config.json", config)
        self.jsonl("errors.jsonl", [{"stage": "sealed_grading", "task_id": "task-1",
            "exception": "RuntimeError('DO_NOT_COPY_HIDDEN_TEST')", "traceback": "DO_NOT_COPY_TRACE"}])
        report = analyze_run(self.run)
        self.assertEqual(report["status"], "no_evaluation_data")
        summary = report["error_summary"]
        self.assertEqual(summary["recorded_error_events"], 2)
        self.assertEqual(summary["stage_counts"], {"sealed_grading": 1, "backend_phase": 1})
        self.assertEqual(summary["type_counts"], {"RuntimeError": 2})
        self.assertIn("errors.jsonl:1", [r["evidence"] for r in summary["records"]])
        self.assertNotIn("DO_NOT_COPY", json.dumps(report))

    def test_durable_breadcrumb_survives_missing_case_and_never_claims_liveness(self):
        self.json('config.json', function_config())
        self.json('current_case.json', {'stage': 'sealed_final_grading', 'state': 'running',
            'task_id': 'task-1', 'pid': 123, 'updated_unix': 42, 'code': 'DO_NOT_COPY_SOURCE'})
        self.jsonl('errors.jsonl', [{'stage': 'generation', 'error_type': 'SystemExit'}])
        report = analyze_run(self.run)
        summary = report['error_summary']
        self.assertEqual(summary['type_counts'], {'SystemExit': 1})
        marker = summary['last_durable_operation']['current_case.json']
        self.assertEqual(marker['stage'], 'sealed_final_grading')
        self.assertTrue(marker['process_liveness_not_checked'])
        self.assertNotIn('DO_NOT_COPY', json.dumps(report))

    def test_superseded_retries_remain_provenance_without_timing_pooling(self):
        self.function_run()
        failed = function_case("dard", case_completed=False, failure_kind="generation_error", generation_wall_s=99)
        rows = [function_case("ar"), function_case("standard"), failed, function_case("dard")]
        self.jsonl("case_results.jsonl", rows)
        self.jsonl("runs.jsonl", [dict(failed, attempt=1, generation={"wall_s": 99}),
                                  dict(function_case("dard"), attempt=1, generation={"wall_s": 4})])
        report = analyze_run(self.run)
        history = report["retry_history"]
        self.assertEqual(history["superseded_result_rows"], 1)
        self.assertEqual(history["superseded_attempt_records"], 1)
        self.assertEqual(history["records"][0]["evidence"], "case_results.jsonl:3")
        self.assertEqual(self.pair(report)["candidate"]["metrics"]["generation_wall_s"], 4)
        self.assertEqual(self.pair(report)["candidate"]["metrics"]["generation_calls"], 1)

    def test_integration_with_actual_function_runner_mock_output(self):
        from test_runner_end2end import FakeBackend, TestEndToEnd
        fixture = TestEndToEnd()
        fixture.setUp()
        original = FakeBackend.metadata
        def full_metadata(backend):
            return {**identity(), **original(backend)}
        try:
            with patch.object(FakeBackend, "metadata", full_metadata):
                self.assertEqual(fixture.run_bench(), 0)
            report = analyze_run(fixture.output)
            self.assertEqual(len(report["cases"]), 18)
            self.assertEqual(len(report["pairs"]), 18)
            self.assertTrue(all(g["eligible_for_controlled_comparison"] for g in report["groups"].values()))
            self.assertTrue(all(c["metrics"]["generation_calls"] == 2 for c in report["cases"]))
            self.assertEqual(report["groups"]["dard_vs_standard/humaneval/py"]["final_outcomes"], {"both_pass": 6})
        finally:
            fixture.doCleanups()

    def test_integration_with_actual_repository_runner_mock_output(self):
        from test_repository_runner import RunnerTests
        fixture = RunnerTests()
        fixture.setUp()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fixture.run_mock()
            report = analyze_run(fixture.output)
            self.assertEqual(len(report["cases"]), 6)
            self.assertEqual(len(report["pairs"]), 6)
            self.assertTrue(all(not g["eligible_for_controlled_comparison"] for g in report["groups"].values()))
            self.assertTrue(all("unknown_final_outcomes" in g["issues"] for g in report["groups"].values()))
            self.assertTrue(all(c["final_pass"] is None for c in report["cases"]))
            self.assertTrue(all(c["metrics"]["generation_calls"] == 4 for c in report["cases"]))
        finally:
            fixture.tearDown()

    def test_multi_run_cli_keeps_incompatible_runs_separate(self):
        self.function_run()
        other = self.run / "other"
        other.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--run", str(self.run), "--run", str(other)]), 0)
        self.assertEqual(json.loads((other / "diagnostics.json").read_text())["status"], "no_evaluation_data")


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def init(self):
        return append_entry(self.run, "init", purpose="Improve solved/hour", hypothesis="Packing cost dominates",
                            change="baseline", split="dev", parent_run=None)

    def test_append_only_chain_intent_observation_decision_and_tamper_detection(self):
        self.init()
        before = (self.run / "experiment_journal.jsonl").read_bytes()
        append_entry(self.run, "observe", text="Fewer forwards but slower", evidence=["diagnostics.json"])
        append_entry(self.run, "decide", text="Profile on dev", next_change="block_size only")
        after = (self.run / "experiment_journal.jsonl").read_bytes()
        self.assertTrue(after.startswith(before))
        self.assertEqual(len(read_journal(self.run)), 3)
        (self.run / "experiment_journal.jsonl").write_bytes(after.replace(b"baseline", b"changed"))
        with self.assertRaisesRegex(ValueError, "integrity"):
            read_journal(self.run)

    def test_no_reinitialization_or_uninitialized_observations(self):
        with self.assertRaisesRegex(ValueError, "Initialize"):
            append_entry(self.run, "observe", text="late intent")
        self.init()
        with self.assertRaisesRegex(ValueError, "already initialized"):
            self.init()

    def test_journal_does_not_mutate_config_and_ignores_evolving_phase_metadata(self):
        config = function_config()
        path = self.run / "config.json"
        path.write_text(json.dumps(config))
        before = path.read_bytes()
        first = self.init()
        self.assertEqual(path.read_bytes(), before)
        updated = copy.deepcopy(config)
        updated["phases"].append({"phase_id": 3})
        path.write_text(json.dumps(updated))
        second = append_entry(self.run, "observe", text="Started")
        self.assertEqual(first["configuration_fingerprints"], second["configuration_fingerprints"])
        self.assertEqual(analyze_run(self.run)["experiment_context"]["split"], "dev")


if __name__ == "__main__":
    unittest.main()
