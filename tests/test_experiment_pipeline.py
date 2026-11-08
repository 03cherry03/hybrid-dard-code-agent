"""The all-experiment planner performs no installations, inference or network I/O."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.experiment_pipeline import build_stages


class ExperimentPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.repo = base / "repository with spaces"
        self.data = base / "data with spaces"
        self.output = base / "output with spaces"
        self.python = "/fixture/python"

    def stages(self, **kwargs):
        return build_stages(repo=self.repo, data_root=self.data, output=self.output,
                            python=self.python, **kwargs)

    @staticmethod
    def option(stage, name):
        argv = stage["command"]
        return argv[argv.index(name) + 1]

    def test_default_functions_scope_has_no_repository_or_library_dependencies(self):
        from datasets_code import CODE_GENERATION_SUITES
        for prepare in (False, True):
            stages = self.stages(prepare=prepare)
            names = [stage["name"] for stage in stages]
            expected = ["gpu_idle_check", "data_verify"]
            if prepare:
                expected += ["setup_functions", "download_models"]
            expected += ["function_smoke", "functions_full", "functions_analysis"]
            self.assertEqual(names, expected)
            self.assertEqual(self.option(stages[1], "--scope"), "functions")
            for stage in stages:
                command = " ".join(stage["command"])
                self.assertNotIn("swebench", command)
                self.assertNotIn("bigcodebench", command)
                self.assertNotIn("system_dependencies", command)
                self.assertNotIn("repository_runner", command)
                if stage["name"] in ("function_smoke", "functions_full"):
                    self.assertEqual(self.option(stage, "--suites").split(","), list(CODE_GENERATION_SUITES))
            if prepare:
                setup = next(stage for stage in stages if stage["name"] == "setup_functions")
                self.assertEqual(setup["command"][-1], "functions")

    def test_repositories_scope_is_explicit_and_omits_function_evaluators(self):
        stages = self.stages(scope="repositories", prepare=True)
        self.assertEqual([stage["name"] for stage in stages], [
            "gpu_idle_check", "data_verify", "repository_data_check", "setup_repositories",
            "download_models", "prepare_verified_repositories", "prepare_multilingual_repositories",
            "repository_smoke", "repository_smoke_grade", "repositories_full", "repositories_grade",
            "repositories_analysis"])
        self.assertEqual(self.option(stages[1], "--scope"), "repositories")
        self.assertEqual(stages[3]["command"][-1], "repositories")
        self.assertFalse(any("--suites" in stage["command"] for stage in stages))

    def test_priority_codegen_calibrates_full_budget_then_runs_ar_dard_before_standard(self):
        deadline = 2_000_000_000.0
        stages = self.stages(priority_codegen=True, deadline_unix=deadline)
        self.assertEqual([stage["name"] for stage in stages], [
            "gpu_idle_check", "data_verify", "function_smoke", "functions_priority",
            "functions_standard", "priority_analysis"])
        indexed = {stage["name"]: stage for stage in stages}
        smoke = indexed["function_smoke"]
        self.assertEqual(self.option(smoke, "--backends"), "ar,dard")
        for flag, value in (("--tokens", "512"), ("--standard-steps", "512"),
                            ("--block-size", "128"), ("--dard-steps", "192"),
                            ("--limit", "2"), ("--repeats", "1"), ("--split", "dev")):
            self.assertEqual(self.option(smoke, flag), value)
        outputs = {self.option(smoke, "--output")}
        for mode in ("priority", "standard"):
            stage = indexed["functions_" + mode]
            self.assertEqual(stage["command"][:3], [self.python,
                str(self.repo / "scripts/priority_experiments.py"), "functions"])
            self.assertEqual(self.option(stage, "--mode"), mode)
            self.assertEqual(float(self.option(stage, "--deadline-unix")), deadline)
            self.assertEqual(self.option(stage, "--calibration"), str(self.output / "function-smoke"))
            self.assertEqual(self.option(stage, "--gpus"), "0,1,2,3")
            self.assertEqual(self.option(stage, "--data-root"), str(self.data))
            self.assertIn("--execute", stage["command"])
            outputs.add(self.option(stage, "--output"))
        self.assertEqual(outputs, {str(self.output / name) for name in
                                  ("function-smoke", "functions-priority", "functions-standard")})
        self.assertIn("--analyze", indexed["priority_analysis"]["command"])
        self.assertFalse(any(stage["downloads"] for stage in stages))
        self.assertFalse(any("standard" == argument for argument in smoke["command"]))

    def test_priority_preparation_finishes_download_before_calibration(self):
        stages = self.stages(priority_codegen=True, deadline_unix=2_000_000_000.0, prepare=True)
        self.assertEqual([stage["name"] for stage in stages], [
            "gpu_idle_check", "data_verify", "setup_functions", "download_models",
            "function_smoke", "functions_priority", "functions_standard", "priority_analysis"])
        self.assertEqual([stage["name"] for stage in stages if stage["downloads"]],
                         ["setup_functions", "download_models"])

    def test_priority_codegen_requires_function_scope_and_absolute_deadline(self):
        for kwargs in ({}, {"scope": "repositories", "deadline_unix": 2_000_000_000.0},
                       {"scope": "all", "deadline_unix": 2_000_000_000.0}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "absolute deadline"):
                self.stages(priority_codegen=True, **kwargs)

    def test_focused_codegen_has_one_required_three_way_campaign_and_analysis(self):
        deadline = 2_000_000_000.0
        stages = self.stages(focused_codegen=True, deadline_unix=deadline)
        self.assertEqual([stage["name"] for stage in stages],
                         ["gpu_idle_check", "data_verify", "focused_codegen", "focused_analysis"])
        generation, analysis = stages[2:]
        self.assertEqual(generation["command"][:3], [self.python,
            str(self.repo / "scripts/focused_experiments.py"), "functions"])
        self.assertEqual(self.option(generation, "--output"), str(self.output / "functions"))
        self.assertEqual(self.option(generation, "--data-root"), str(self.data))
        self.assertEqual(self.option(generation, "--gpus"), "0,1,2,3")
        self.assertEqual(float(self.option(generation, "--deadline-unix")), deadline)
        self.assertIn("--execute", generation["command"])
        self.assertEqual(self.option(analysis, "--output"), self.option(generation, "--output"))
        self.assertIn("--analyze", analysis["command"])
        self.assertNotIn("--execute", analysis["command"])
        self.assertFalse(any(stage["downloads"] for stage in stages))
        for stage in stages:
            self.assertNotIn("--repeats", stage["command"])
            self.assertNotIn("--mode", stage["command"])
            self.assertNotIn("function_smoke", stage["name"])
            self.assertNotIn("priority", " ".join(stage["command"]))

    def test_focused_preparation_adds_only_required_setup_and_download(self):
        stages = self.stages(focused_codegen=True, deadline_unix=2_000_000_000.0, prepare=True)
        self.assertEqual([stage["name"] for stage in stages], [
            "gpu_idle_check", "data_verify", "setup_functions", "download_models",
            "focused_codegen", "focused_analysis"])

    def test_focused_codegen_rejects_incompatible_scope_missing_deadline_and_priority(self):
        for kwargs in ({}, {"scope": "repositories", "deadline_unix": 2_000_000_000.0},
                       {"scope": "all", "deadline_unix": 2_000_000_000.0},
                       {"priority_codegen": True, "deadline_unix": 2_000_000_000.0}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "focused code generation"):
                self.stages(focused_codegen=True, **kwargs)

    def test_explicit_all_stage_sequence_has_no_downloads(self):
        stages = self.stages(scope="all")
        self.assertEqual([stage["name"] for stage in stages], [
            "gpu_idle_check", "data_verify", "repository_data_check",
            "function_smoke", "repository_smoke", "repository_smoke_grade",
            "functions_full", "repositories_full", "repositories_grade",
            "functions_analysis", "repositories_analysis"])
        self.assertFalse(any(stage["downloads"] for stage in stages))
        self.assertFalse(any("--pull" in stage["command"] or "--download" in stage["command"]
                             for stage in stages))

    def test_explicit_preparation_finishes_all_local_environments_before_inference(self):
        stages = self.stages(prepare=True, scope="all")
        names = [stage["name"] for stage in stages]
        self.assertEqual(names[3:7], ["setup_all", "download_models", "prepare_verified_repositories",
                                     "prepare_multilingual_repositories"])
        self.assertEqual([stage["name"] for stage in stages if stage["downloads"]], names[3:7])
        self.assertLess(names.index("prepare_multilingual_repositories"), names.index("function_smoke"))
        models = next(stage for stage in stages if stage["name"] == "download_models")
        self.assertEqual(models["command"][0], str(self.data / "envs/inference/bin/python"))
        for name, suite in (("prepare_verified_repositories", "swebench_verified"),
                            ("prepare_multilingual_repositories", "swebench_multilingual")):
            stage = next(stage for stage in stages if stage["name"] == name)
            self.assertEqual(self.option(stage, "--suite"), suite)
            self.assertEqual(self.option(stage, "--limit"), "0")
            self.assertIn("--prepare", stage["command"])
            self.assertEqual(stage["command"][1], str(self.repo / "scripts/prepare_repositories.py"))

    def test_all_generation_stages_keep_paired_four_gpu_offline_summary_protocol(self):
        stages = self.stages(gpus="GPU-a,GPU-b,GPU-c,GPU-d", scope="all")
        runs = [stage for stage in stages if "--execute" in stage["command"]]
        self.assertEqual(len(runs), 4)
        for stage in runs:
            with self.subTest(stage=stage["name"]):
                self.assertEqual(self.option(stage, "--gpus"), "GPU-a,GPU-b,GPU-c,GPU-d")
                self.assertEqual(self.option(stage, "--data-root"), str(self.data))
                self.assertEqual(self.option(stage, "--seed"), "42")
                self.assertEqual(self.option(stage, "--quant"), "bf16")
                self.assertEqual(self.option(stage, "--diagnostics"), "summary")
                self.assertIn("--fail-fast", stage["command"])

    def test_full_function_and_repository_budgets_are_fixed_and_distinct(self):
        indexed = {stage["name"]: stage for stage in self.stages(scope="all")}
        for name, tokens, steps in (("functions_full", "512", "512"),
                                   ("repositories_full", "1024", "128")):
            stage = indexed[name]
            self.assertEqual(self.option(stage, "--limit"), "0")
            self.assertEqual(self.option(stage, "--repeats"), "3")
            self.assertEqual(self.option(stage, "--tokens"), tokens)
            self.assertEqual(self.option(stage, "--standard-steps"), steps)
            self.assertEqual(self.option(stage, "--block-size"), "128")
            self.assertEqual(self.option(stage, "--dard-steps"), "192")
            self.assertEqual(self.option(stage, "--split"), "test")
        self.assertEqual(self.option(indexed["repositories_full"], "--task-timeout"), "600")
        self.assertEqual(self.option(indexed["repositories_full"], "--max-steps"), "30")

    def test_smoke_cases_fill_four_shards_and_use_separate_output_paths(self):
        indexed = {stage["name"]: stage for stage in self.stages(scope="all")}
        self.assertEqual(self.option(indexed["function_smoke"], "--limit"), "1")
        self.assertEqual(self.option(indexed["repository_smoke"], "--limit"), "2")
        for name in ("function_smoke", "repository_smoke"):
            self.assertEqual(self.option(indexed[name], "--repeats"), "1")
            self.assertEqual(self.option(indexed[name], "--split"), "dev")
        outputs = [self.option(stage, "--output") for stage in indexed.values()
                   if "--execute" in stage["command"]]
        self.assertEqual(len(set(outputs)), 4)
        self.assertTrue(all(Path(output).parent == self.output for output in outputs))

    def test_postprocessing_uses_existing_plans_after_generation(self):
        stages = self.stages(scope="all")
        names = [stage["name"] for stage in stages]
        indexed = {stage["name"]: stage for stage in stages}
        for post, generation, action in (
                ("repository_smoke_grade", "repository_smoke", "--grade"),
                ("repositories_grade", "repositories_full", "--grade"),
                ("functions_analysis", "functions_full", "--analyze"),
                ("repositories_analysis", "repositories_full", "--analyze")):
            self.assertGreater(names.index(post), names.index(generation))
            self.assertEqual(self.option(indexed[post], "--output"), self.option(indexed[generation], "--output"))
            self.assertEqual(indexed[post]["command"][-1], action)
            self.assertNotIn("--execute", indexed[post]["command"])
            self.assertNotIn("--tokens", indexed[post]["command"])
            self.assertNotIn("--gpus", indexed[post]["command"])
        self.assertGreater(names.index("repositories_analysis"), names.index("repositories_grade"))

    def test_data_checks_preserve_original_task_files_and_root(self):
        indexed = {stage["name"]: stage for stage in self.stages(scope="all")}
        self.assertEqual(indexed["gpu_idle_check"]["command"], [self.python,
            str(self.repo / "scripts/all_experiments.py"), "--check-gpus", "--gpus", "0,1,2,3"])
        verify = indexed["data_verify"]["command"]
        self.assertEqual(verify[0], self.python)
        self.assertIn("--verify", verify)
        self.assertEqual(self.option(indexed["data_verify"], "--data-root"), str(self.data))
        checks = indexed["repository_data_check"]["command"]
        self.assertIn(str(self.data / "datasets/swebench_verified/tasks.jsonl"), checks)
        self.assertIn(str(self.data / "datasets/swebench_multilingual/tasks.jsonl"), checks)

    def test_planning_has_no_subprocess_or_filesystem_write_side_effects(self):
        with patch("subprocess.run", side_effect=AssertionError("Planner executed a command")), \
             patch("subprocess.Popen", side_effect=AssertionError("Planner started a process")):
            stages = self.stages(prepare=True)
        self.assertTrue(stages)
        self.assertFalse(self.repo.exists())
        self.assertFalse(self.data.exists())
        self.assertFalse(self.output.exists())
        for stage in stages:
            self.assertEqual(stage["cwd"], str(self.repo))
            self.assertTrue(all(isinstance(arg, str) for arg in stage["command"]))

    def test_plan_is_deterministic_and_returns_independent_mutable_records(self):
        first, second = self.stages(prepare=True), self.stages(prepare=True)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len({stage["name"] for stage in first}))
        first[0]["command"].append("changed")
        self.assertNotEqual(first, second)

    def test_invalid_gpu_selection_and_preparation_flag_are_rejected(self):
        for gpus in ("0,1,2", "0,1,1,3", "0,1,,3", None):
            with self.subTest(gpus=gpus), self.assertRaises(ValueError):
                self.stages(gpus=gpus)
        with self.assertRaises(ValueError):
            self.stages(prepare="yes")
        for scope in ("function", "library", "", None):
            with self.subTest(scope=scope), self.assertRaisesRegex(ValueError, "scope"):
                self.stages(scope=scope)
        self.assertEqual(self.option(next(stage for stage in self.stages(gpus="0, 1, 2, 3")
                                          if stage["name"] == "function_smoke"), "--gpus"), "0,1,2,3")


class SetupScopeTests(unittest.TestCase):
    def setUp(self):
        self.script = (Path(__file__).resolve().parents[1] / "scripts/setup_env.sh").read_text()

    def test_setup_dispatch_only_calls_selected_environment_builders(self):
        # Execute the production dispatch with inert builders, never installers.
        dispatch = self.script[self.script.rindex('case "$task_target" in'):]
        dispatch = dispatch[:dispatch.index("\nesac") + len("\nesac")]
        stubs = "\n".join(f"setup_{name}() {{ echo {name}; }}" for name in
                          ("inference", "function", "swebench", "evaluators"))
        expected = {"functions": ["inference", "function", "evaluators"],
                    "repositories": ["inference", "function", "swebench"],
                    "all": ["inference", "function", "swebench", "evaluators"],
                    "evaluators": ["evaluators"]}
        for scope, names in expected.items():
            result = subprocess.run(["bash", "--noprofile", "--norc", "-c",
                                     stubs + '\ntask_target="$1"\n' + dispatch, "fixture", scope],
                                    capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.splitlines(), names)

    def test_system_dependency_check_cannot_run_for_function_scopes(self):
        start = self.script.index('if [[ "$task_target"')
        guard = self.script[start:self.script.index("\nfi", start) + len("\nfi")]
        fixture = ('python3() { echo SYSTEM_CHECK; }\n'
                   'task_script_dir=/fixture\nHYBRID_DARD_ROOT=/data/fixture\n'
                   'task_target="$1"\n')
        for scope in ("functions", "function", "inference", "evaluators", "repositories", "swebench", "all"):
            result = subprocess.run(["bash", "--noprofile", "--norc", "-c", fixture + guard, "fixture", scope],
                                    capture_output=True, text=True, check=True)
            expected = "SYSTEM_CHECK" if scope in ("repositories", "swebench", "all") else ""
            self.assertEqual(result.stdout.strip(), expected)
        self.assertNotIn("bigcodebench", self.script.lower())


if __name__ == "__main__":
    unittest.main()
