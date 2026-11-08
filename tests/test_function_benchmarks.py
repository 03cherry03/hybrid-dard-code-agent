"""Offline official evaluator fixtures: no package installation or dataset downloads."""
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import datasets_code as data
import evaluation
import function_benchmarks as official
import official_function_worker as worker


def fixture(suite="humaneval_plus", number=0):
    if suite == "bigcodebench_instruct":
        return {"task_id": f"BigCodeBench/{number}", "instruct_prompt": "Implement def f(x): identity.",
                "complete_prompt": "def f(x):\n    '''identity'''\n", "code_prompt": "def f(x):",
                "canonical_solution": "    return x # SECRET_SOLUTION\n",
                "test": "SECRET_SEALED_UNITTEST", "entry_point": "f"}
    mbpp = suite == "mbpp_plus"
    return {"task_id": f"{'Mbpp' if mbpp else 'HumanEval'}/{number}",
            "prompt": '"""Return x. Example: f(1) == 1."""\n' if mbpp else
                      "from typing import List\n\ndef f(x):\n    '''Return x.'''\n",
            "entry_point": "f", "canonical_solution":
            "def helper(): return 'SECRET_HELPER'\ndef f(x): return x # SECRET_SOLUTION" if mbpp else
            "    return x # SECRET_SOLUTION\n",
            "contract": "SECRET_CONTRACT", "base_input": [[1]], "plus_input": [[987654321]], "atol": 0}


class _FixtureWorkspace:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def prepare(self, suite, rows=None, gz=False):
        path = official.manifest_path(self.root).parent / "raw" / (suite + ".jsonl" + (".gz" if gz else ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = "\n".join(json.dumps(row) for row in (rows or [fixture(suite)])).encode()
        path.write_bytes(gzip.compress(payload) if gz else payload)
        official.write_official_manifest(self.root, {suite: path})
        return path


class ManifestTests(_FixtureWorkspace, unittest.TestCase):
    def test_three_official_adapters_keep_oracles_out_of_task_and_prompt(self):
        for suite in official.OFFICIAL_SUITES:
            with self.subTest(suite=suite):
                self.prepare(suite, gz=True)
                task = official.load_official_tasks(suite, self.root)[0]
                encoded = json.dumps(asdict(task))
                self.assertNotIn("SECRET", encoded)
                self.assertNotIn("987654321", encoded)
                self.assertEqual(task.test_source, "")
                self.assertEqual(task.feedback_test_source, "")
                self.assertEqual(task.evaluator_metadata["dataset_task_count"], 1)
                self.assertIn("f(x)", task.prompt)
                self.assertEqual(official._official_record(task), fixture(suite))
                if suite == "mbpp_plus":
                    self.assertIn("def f(x):", task.prompt)
                    self.assertNotIn("helper", task.prompt)

    def test_mbpp_plus_keeps_official_scope_instead_of_native_id_filter(self):
        self.prepare("mbpp_plus", [fixture("mbpp_plus", 2), fixture("mbpp_plus", 809)])
        tasks = data.load_tasks(["mbpp_plus"], self.root / "unused", 0, data_root=self.root)
        self.assertEqual([task.number for task in tasks], [2, 809])
        self.assertEqual(tasks[1].source_id, "MBPP/809")

    def test_changed_raw_file_rejected_before_grading(self):
        raw = self.prepare("humaneval_plus")
        task = official.load_official_tasks("humaneval_plus", self.root)[0]
        raw.write_text(raw.read_text() + "\n ", encoding="utf-8")
        with patch.object(official, "run_bounded_command") as run:
            result = official.grade_official(task, "def f(x): return x")
        self.assertIsNone(result["passed"])
        self.assertIn("checksum mismatch", result["sealed_diagnostics"])
        run.assert_not_called()

    def test_duplicates_and_unsupported_evaluator_versions_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.prepare("humaneval_plus", [fixture(), fixture()])
        self.prepare("humaneval_plus")
        path = official.manifest_path(self.root)
        doc = json.loads(path.read_text())
        doc["suites"]["humaneval_plus"]["evaluator_version"] = "future"
        path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, "Unsupported.*version"):
            official.load_official_tasks("humaneval_plus", self.root)

    def test_missing_legacy_and_official_data_never_fetch_implicitly(self):
        with patch.object(data.urllib.request, "urlopen") as fetch:
            with self.assertRaises(FileNotFoundError):
                data.load_humaneval(self.root / "missing")
            with self.assertRaises(FileNotFoundError):
                official.load_official_tasks("humaneval_plus", self.root)
        fetch.assert_not_called()


class SeparationTests(unittest.TestCase):
    def test_docstring_examples_are_taken_only_from_public_prompt(self):
        prompt = "def f(x):\n    '''Return x.\n\n    >>> f(123)\n    123\n    '''\n"
        tests = data.public_feedback_tests(prompt, "f")
        self.assertIn("f(123)", tests)
        self.assertIn("123", tests)
        self.assertIn("DocTestRunner", tests)
        self.assertIn("failures == 0", tests)
        self.assertEqual(data.public_feedback_tests("This flattened prose >>> f(123) 123", "f"), "")

    def test_mbpp_assertion_examples_are_public_and_do_not_need_reference(self):
        prompt = '\"\"\"Return x.\nassert f(3) == 3\nassert f(\n  4\n) == 4\n\"\"\"'
        tests = data.public_feedback_tests(prompt, "f")
        self.assertEqual(tests, "assert f(3) == 3\nassert f(4) == 4\n")

    def test_incomplete_comparison_doctest_cannot_falsely_reject_correct_source(self):
        prompt = "def f(x):\n    '''Return x.\n    >>> f(1) == 1\n    '''\n"
        self.assertEqual(data.public_feedback_tests(prompt, "f"), "")
        complete = "def f(x):\n    '''Return x.\n    >>> f(1) == 1\n    True\n    '''\n"
        self.assertIn("DocTestRunner", data.public_feedback_tests(complete, "f"))

    def test_public_examples_use_official_dependency_environment(self):
        task = data.Task("fixture", "py", "0", 0, "", "SEALED_ORACLE",
                         feedback_test_source="assert f(1) == 1", final_evaluator="bigcodebench")
        def simulate(command, **kwargs):
            self.assertEqual(command[0], "/prepared/bigcodebench/bin/python")
            source = (Path(kwargs["cwd"]) / "solution.py").read_text()
            self.assertIn("assert f(1) == 1", source)
            self.assertNotIn("SEALED", source)
            return SimpleNamespace(returncode=0, output="", timed_out=False)
        with patch.object(evaluation, "run_bounded_command", side_effect=simulate), \
             patch.object(evaluation, "official_python", return_value="/prepared/bigcodebench/bin/python"):
            verdict = evaluation.test_task(task, "def f(x): return x", "local")
        self.assertEqual(verdict["feedback_scope"], "public_tests")

    def test_official_compile_only_feedback_uses_its_target_python_version(self):
        task = data.Task("fixture", "py", "0", 0, "", "SEALED_ORACLE", final_evaluator="bigcodebench")
        def simulate(command, **kwargs):
            self.assertEqual(command[0], "/prepared/bigcodebench/bin/python")
            self.assertIn("ast.parse", command[-2])
            self.assertTrue(kwargs["env"]["NLTK_DATA"].endswith("/cache/nltk_data"))
            return SimpleNamespace(returncode=0, output="", timed_out=False)
        with patch.object(evaluation, "run_bounded_command", side_effect=simulate), \
             patch.object(evaluation, "official_python", return_value="/prepared/bigcodebench/bin/python"):
            verdict = evaluation.test_task(task, "def f(x): return x", "local")
        self.assertEqual(verdict["feedback_scope"], "syntax_compile_only")

    def test_feedback_programs_never_include_final_tests(self):
        samples = {"py": "def f(x): return x", "js": "function f(x) { return x; }",
                   "rs": "fn f(x:i64)->i64 {x}", "cpp": "long f(long x) {return x;}"}
        for lang, code in samples.items():
            task = data.Task("fixture", lang, "0", 0, "", "SEALED_ORACLE_SENTINEL")
            source, command = evaluation._program(task, code, stage="feedback")
            self.assertNotIn("SEALED", source)
            self.assertNotIn("exec ", command[-1])
            if lang == "py":
                self.assertIn("ast.parse", command[-2])
            elif lang == "js":
                self.assertIn("--check", command)
            elif lang == "rs":
                self.assertIn("--crate-type lib", command[-1])
            else:
                self.assertIn("-fsyntax-only", command[-1])

    def test_explicit_public_tests_only_run_in_feedback_stage(self):
        task = data.Task("fixture", "py", "0", 0, "", "assert f(987654321) == 1 # SEALED",
                         feedback_test_source="assert f(1) == 1 # PUBLIC")
        source, command = evaluation._program(task, "def f(x): return x", stage="feedback")
        self.assertIn("PUBLIC", source)
        self.assertNotIn("SEALED", source)
        self.assertEqual(command[-1], str(Path.cwd() / "solution.py"))

    def test_final_diagnostics_cannot_enter_repair_feedback(self):
        task = data.Task("fixture", "py", "0", 0, "", "assert False # SECRET")
        with patch.object(evaluation, "local_executable", return_value=sys.executable), \
             patch.object(evaluation, "run_bounded_command", return_value=SimpleNamespace(
                 returncode=1, output="AssertionError: SECRET expected answer", timed_out=False)):
            result = evaluation.grade_final(task, "def f(): pass", "local")
        self.assertFalse(result["passed"])
        self.assertEqual(result["feedback"], "")
        self.assertTrue(result["sealed"])
        self.assertIn("SECRET", result["sealed_diagnostics"])

    def test_syntax_success_is_labeled_feedback_only(self):
        task = data.Task("fixture", "py", "0", 0, "", "assert False")
        with patch.object(evaluation, "local_executable", return_value=sys.executable), \
             patch.object(evaluation, "run_bounded_command", return_value=SimpleNamespace(
                 returncode=0, output="", timed_out=False)):
            result = evaluation.test_task(task, "def f(): pass", "local")
        self.assertTrue(result["passed"])
        self.assertEqual(result["status"], "feedback_passed")
        self.assertEqual(result["feedback_scope"], "syntax_compile_only")
        self.assertFalse(result["sealed"])


class OfficialRunnerTests(_FixtureWorkspace, unittest.TestCase):
    def test_local_worker_receives_fresh_snapshot_and_explicit_paths(self):
        self.prepare("humaneval_plus")
        task = official.load_official_tasks("humaneval_plus", self.root)[0]
        paths = []
        def simulate(command, **kwargs):
            self.assertEqual(command[0], "/prepared/evalplus/bin/python")
            work = Path(command[command.index("--work") + 1])
            scratch = Path(command[command.index("--scratch") + 1])
            paths.extend([work, scratch])
            raw = json.loads((work / "raw.jsonl").read_text())
            request = json.loads(Path(command[command.index("--request") + 1]).read_text())
            self.assertEqual(raw, fixture())
            self.assertNotIn("SECRET", request["solution"])
            self.assertIn("from typing import List", request["solution"])
            self.assertEqual(kwargs["cwd"], scratch)
            self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
            return SimpleNamespace(returncode=0, output=json.dumps(
                {"task_id": task.task_id, "passed": True, "status": "passed"}),
                timed_out=False, output_truncated=False)
        with patch.object(evaluation, "official_python", return_value="/prepared/evalplus/bin/python"), \
             patch.object(official, "run_bounded_command", side_effect=simulate):
            result = evaluation.grade_final(task, "def f(x): return x", "local")
        self.assertTrue(result["passed"])
        self.assertEqual(result["feedback"], "")
        self.assertTrue(all(not path.exists() for path in paths))

    def test_whole_grader_timeout_is_infrastructure(self):
        self.prepare("humaneval_plus")
        task = official.load_official_tasks("humaneval_plus", self.root)[0]
        with patch.object(evaluation, "official_python", return_value=sys.executable), \
             patch.object(official, "run_bounded_command", return_value=SimpleNamespace(
                returncode=-9, output="", timed_out=True, output_truncated=False)) as run:
            result = evaluation.grade_final(task, "def f(x): return x", "local", timeout=1)
        self.assertIsNone(result["passed"])
        self.assertEqual(result["status"], "infra_error")
        self.assertTrue(result["sealed_diagnostics"])
        self.assertEqual(result["execution"]["signal"], 9)
        self.assertEqual(result["execution"]["returncode"], -9)
        self.assertTrue(result["execution"]["timed_out"])
        self.assertEqual(result["feedback"], "")
        self.assertEqual(run.call_count, 1)

    def test_old_unused_manifest_metadata_does_not_change_native_task(self):
        self.prepare("humaneval_plus")
        original = official.load_official_tasks("humaneval_plus", self.root)[0]
        path = official.manifest_path(self.root)
        doc = json.loads(path.read_text())
        doc["suites"]["humaneval_plus"]["evaluator_image"] = "legacy-unused"
        path.write_text(json.dumps(doc))
        self.assertEqual(official.load_official_tasks("humaneval_plus", self.root)[0], original)
        self.assertNotIn("evaluator_image", original.evaluator_metadata)

    def test_preflight_pins_package_versions_in_separate_environments(self):
        self.prepare("humaneval_plus")
        task = official.load_official_tasks("humaneval_plus", self.root)[0]
        for name in ("function", "evalplus"):
            interpreter = self.root / "envs" / name / "bin/python"
            interpreter.parent.mkdir(parents=True)
            interpreter.symlink_to(sys.executable)
        def probe(command, **kwargs):
            packages = [["evalplus", "0.3.1"]] if "evalplus" in command[0] else []
            return SimpleNamespace(returncode=0, timed_out=False, output_truncated=False,
                output=json.dumps({"executable": command[0], "python": "3.fixture", "packages": packages}))
        with patch.object(evaluation, "_LOCAL_DATA_ROOT", None), \
             patch.dict(evaluation._EXECUTABLES, {}, clear=True), \
             patch.dict(evaluation._OFFICIAL_PYTHONS, {}, clear=True), \
             patch.object(evaluation, "run_bounded_command", side_effect=probe):
            identities = evaluation.preflight_local({"py"}, tasks=[task], data_root=self.root)
            self.assertEqual(identities["official:evalplus"]["evaluator_version"], "0.3.1")
            self.assertIn("envs/evalplus/bin/python", evaluation.official_python("evalplus"))
        with patch.object(evaluation, "run_bounded_command", return_value=SimpleNamespace(
                returncode=0, timed_out=False, output_truncated=False,
                output=json.dumps({"python": "3.fixture", "packages": [["evalplus", "wrong"]]}))):
            with self.assertRaisesRegex(RuntimeError, "version mismatch"):
                evaluation.preflight_local({"py"}, tasks=[task], data_root=self.root)

    def test_inconsistent_or_wrong_task_worker_result_is_infrastructure(self):
        self.prepare("humaneval_plus")
        task = official.load_official_tasks("humaneval_plus", self.root)[0]
        for answer in ({"task_id": "wrong", "passed": True, "status": "passed"},
                       {"task_id": task.task_id, "passed": True, "status": "infra_error"}):
            with patch.object(evaluation, "official_python", return_value=sys.executable), \
                 patch.object(official, "run_bounded_command", return_value=SimpleNamespace(
                     returncode=0, output=json.dumps(answer), timed_out=False, output_truncated=False)):
                result = official.grade_official(task, "def f(x): return x")
            self.assertIsNone(result["passed"])
            self.assertEqual(result["status"], "infra_error")


class OfficialWorkerTests(unittest.TestCase):
    def test_official_commands_use_full_oracle_and_local_bcb_execution(self):
        command = worker.official_command("evalplus", "mbpp", "samples.jsonl")
        self.assertIn("evalplus.evaluate", command)
        self.assertNotIn("--base-only", command)
        self.assertNotIn("--mini", command)
        bcb = worker.official_command("bigcodebench", "instruct", "samples.jsonl")
        self.assertEqual(bcb[bcb.index("--execution") + 1], "local")
        self.assertNotIn("--no_gt", bcb)
        self.assertEqual(bcb[bcb.index("--calibrated") + 1], "False")

    def test_evalplus_requires_both_base_and_extra_tests(self):
        def doc(base, plus):
            return {"eval": {"HumanEval/0": [{"task_id": "HumanEval/0", "base_status": base,
                                                "plus_status": plus, "plus_fail_tests": ["SECRET"]}]}}
        result = worker.parse_official_result(doc("pass", "fail"), "evalplus", "HumanEval/0")
        self.assertFalse(result["passed"])
        self.assertNotIn("SECRET", json.dumps(result))
        with self.assertRaises(ValueError):
            worker.parse_official_result(doc("pass", None), "evalplus", "HumanEval/0")

    def test_bcb_reference_environment_failure_is_not_candidate_failure(self):
        doc = {"eval": {"BigCodeBench/0": [{"task_id": "BigCodeBench/0", "status": "pass"}]}}
        failed = worker.parse_official_result(doc, "bigcodebench", "BigCodeBench/0",
                                              groundtruth={"gt_pass_rate": 0, "failed_tasks": ["BigCodeBench/0"]})
        self.assertIsNone(failed["passed"])
        passed = worker.parse_official_result(doc, "bigcodebench", "BigCodeBench/0",
                                              groundtruth={"gt_pass_rate": 1, "failed_tasks": []})
        self.assertTrue(passed["passed"])

    def test_worker_sets_official_override_and_checks_version_without_real_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            request = {"task_id": "HumanEval/0", "evaluator": "evalplus", "version": "0.3.1",
                       "dataset": "humaneval", "solution": "def f(): return 1"}
            def simulate(candidate, **kwargs):
                self.assertEqual(candidate, request)
                self.assertEqual(kwargs["env"]["HUMANEVAL_OVERRIDE_PATH"], str(work / "raw.jsonl"))
                doc = {"eval": {"HumanEval/0": [{"task_id": "HumanEval/0", "base_status": "pass", "plus_status": "pass"}]}}
                return worker.parse_official_result(doc, "evalplus", "HumanEval/0")
            with patch.object(worker.importlib.metadata, "version", return_value="0.3.1"), \
                 patch.object(worker, "_run_evalplus_api", side_effect=simulate):
                result = worker.run_request(request, work, work)
            self.assertTrue(result["passed"])
            self.assertNotIn("SECRET", json.dumps(result))
            with patch.object(worker.importlib.metadata, "version", return_value="wrong"), \
                 patch.object(worker, "_run_evalplus_api") as run:
                with self.assertRaisesRegex(RuntimeError, "version mismatch"):
                    worker.run_request(request, work, work)
                run.assert_not_called()

    def test_evalplus_api_preserves_official_input_loader_special_oracle_and_defaults(self):
        from unittest.mock import Mock
        problems = {"Mbpp/2": {"task_id": "Mbpp/2", "base_input": [(1, 2)]}}
        expected = {"Mbpp/2": {"official_oracle": "fixture"}}
        special = ["check_str", "text_match_three", "text_starta_endb"]
        loaders = SimpleNamespace(get_human_eval_plus=Mock(), get_human_eval_plus_hash=Mock(),
                                  get_mbpp_plus=Mock(return_value=problems),
                                  get_mbpp_plus_hash=Mock(return_value="official-hash"))
        checks = SimpleNamespace(get_groundtruth=Mock(return_value=expected),
                                 check_correctness=Mock(return_value={"task_id": "Mbpp/2",
                                     "base": ("pass", [True]), "plus": ("fail", [False])}))
        modules = {"evalplus.data": loaders, "evalplus.evaluate": checks,
                   "evalplus.eval._special_oracle": SimpleNamespace(MBPP_OUTPUT_NOT_NONE_TASKS=special)}
        request = {"dataset": "mbpp", "task_id": "Mbpp/2", "solution": "candidate"}
        with patch.dict(sys.modules, modules), patch.dict(worker.os.environ, {}, clear=True):
            result = worker._run_evalplus_api(request, env={"MBPP_OVERRIDE_PATH": "/fresh/raw.jsonl"})
            self.assertNotIn("MBPP_OVERRIDE_PATH", worker.os.environ)
        loaders.get_mbpp_plus.assert_called_once_with()
        checks.get_groundtruth.assert_called_once_with(problems, "official-hash", special)
        checks.check_correctness.assert_called_once_with("mbpp", 0, problems["Mbpp/2"], "candidate",
                                                        expected["Mbpp/2"], base_only=False, fast_check=True)
        self.assertEqual(result["official_statuses"], ["pass", "fail"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["feedback"], "")
        checks.get_groundtruth.reset_mock()
        problems["Mbpp/3"] = {"task_id": "Mbpp/3"}
        with patch.dict(sys.modules, modules), patch.dict(worker.os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "exactly the submitted task"):
                worker._run_evalplus_api(request, env={"MBPP_OVERRIDE_PATH": "/fresh/raw.jsonl"})
        checks.get_groundtruth.assert_not_called()


if __name__ == "__main__":
    unittest.main()
